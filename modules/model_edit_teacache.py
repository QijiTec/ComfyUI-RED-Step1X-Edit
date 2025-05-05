import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .connector_edit import Qwen2Connector
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock
from ..utils.teacache import TeaCacheForStep1XEdit


class Step1XEditWithTeaCache(nn.Module):
    """
    Transformer model for flow matching on sequences with TeaCache acceleration.
    """

    def __init__(self, params, rel_l1_thresh=0.6, verbose=False):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.connector = Qwen2Connector()

        # Initialize TeaCache
        self.teacache = TeaCacheForStep1XEdit(rel_l1_thresh=rel_l1_thresh, verbose=verbose)

    @staticmethod
    def timestep_embedding(
        t: Tensor, dim, max_period=10000, time_factor: float = 1000.0
    ):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Check if we should compute the transformer blocks or reuse cached results
        # Get normalized input for TeaCache detection using first double block's layernorm
        norm1_output = self.double_blocks[0].img_norm1(img)  # This is now a LayerNorm layer

        if self.teacache.enable_teacache:
            should_calc = self.teacache.should_calculate(norm1_output)

            if not should_calc and self.teacache.previous_residual is not None:
                # Reuse previous residual
                img = img + self.teacache.previous_residual
            else:
                # Compute new residual
                ori_img = img.clone()

                # Process through double blocks
                for block in self.double_blocks:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

                # Concatenate and process through single blocks
                img = torch.cat((txt, img), 1)
                for block in self.single_blocks:
                    img = block(img, vec=vec, pe=pe)
                img = img[:, txt.shape[1]:, ...]

                # Store the residual for future reuse
                if self.teacache.enable_teacache:
                    self.teacache.store_residual(ori_img, img)
        else:
            # Standard processing without TeaCache
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            img = torch.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
            img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def reset_teacache(self, num_steps):
        """Reset TeaCache state for a new denoising process."""
        self.teacache.reset(num_steps)

    def set_teacache_enabled(self, enabled):
        """Enable or disable TeaCache acceleration."""
        self.teacache.enable_teacache = enabled

    def set_teacache_threshold(self, threshold):
        """Set the threshold for TeaCache."""
        self.teacache.rel_l1_thresh = threshold