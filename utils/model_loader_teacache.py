import torch
from .step1x_edit_utils import load_state_dict
from ..modules.autoencoder import AutoEncoder
from ..modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from ..modules.model_edit import Step1XParams
from ..modules.model_edit_teacache import Step1XEditWithTeaCache


class Step1XEditTeaCacheModelBundle:
    """Bundle containing all components of the Step1X-Edit model with TeaCache acceleration."""

    def __init__(self, ae=None, dit=None, llm_encoder=None, device="cuda", dtype=torch.bfloat16,
                 rel_l1_thresh=0.6, verbose=False):
        self.ae = ae
        self.dit = dit
        self.llm_encoder = llm_encoder
        self.device = device
        self.dtype = dtype
        self.quantized = False
        self.offload = False
        self.rel_l1_thresh = rel_l1_thresh
        self.verbose = verbose

    def load_models(self, dit_path, ae_path, qwen2vl_model_path, max_length=256):
        print(f"Loading Step1X-Edit models with TeaCache acceleration (rel_l1_thresh={self.rel_l1_thresh})...")
        print(f"Model paths: \n - DIT: {dit_path}, \n - VAE: {ae_path}, \n - Qwen2VL: {qwen2vl_model_path}")

        """Load all model components with TeaCache acceleration."""
        if Step1XParams is None or AutoEncoder is None or Qwen2VLEmbedder is None:
            print("Error: Step1X-Edit modules not found. Please make sure they are properly installed.")
            return False

        # Initialize LLM encoder
        self.llm_encoder = Qwen2VLEmbedder(
            qwen2vl_model_path,
            device=self.device,
            max_length=max_length,
            dtype=self.dtype,
        )

        # Initialize autoencoder
        with torch.device("meta"):
            self.ae = AutoEncoder(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            )

            # Initialize DIT model with TeaCache support
            step1x_params = Step1XParams(
                in_channels=64,
                out_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
            )

            # Initialize TeaCache-enabled DIT model
            self.dit = Step1XEditWithTeaCache(
                params=step1x_params,
                rel_l1_thresh=self.rel_l1_thresh,
                verbose=self.verbose
            )

        # Load weights
        self.ae = load_state_dict(self.ae, ae_path, 'cpu')
        self.dit = load_state_dict(self.dit, dit_path, 'cpu')

        # Set model precision
        self.ae = self.ae.to(dtype=torch.float32)

        if not self.quantized:
            self.dit = self.dit.to(dtype=self.dtype)

        if not self.offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)

        print("Step1X-Edit models with TeaCache acceleration loaded successfully.")
        return True

    def set_quantized(self, quantized):
        """Set whether to use quantized weights."""
        self.quantized = quantized

    def set_offload(self, offload):
        """Set whether to offload models to CPU."""
        self.offload = offload

    def to_device(self, device):
        """Move models to specified device."""
        self.device = device
        if not self.offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)

    def set_teacache_threshold(self, threshold):
        """Set the TeaCache threshold."""
        self.rel_l1_thresh = threshold
        if hasattr(self.dit, 'set_teacache_threshold'):
            self.dit.set_teacache_threshold(threshold)

    def enable_teacache(self, enabled=True):
        """Enable or disable TeaCache acceleration."""
        if hasattr(self.dit, 'set_teacache_enabled'):
            self.dit.set_teacache_enabled(enabled)

    def reset_teacache(self, num_steps):
        """Reset TeaCache state for a new denoising process."""
        if hasattr(self.dit, 'reset_teacache'):
            self.dit.reset_teacache(num_steps)

    def get_models(self):
        """Return all model components."""
        return {
            "ae": self.ae,
            "dit": self.dit,
            "llm_encoder": self.llm_encoder,
            "device": self.device,
            "dtype": self.dtype,
            "quantized": self.quantized,
            "offload": self.offload,
            "rel_l1_thresh": self.rel_l1_thresh,
            "verbose": self.verbose
        }