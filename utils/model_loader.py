import torch
from .step1x_edit_utils import load_state_dict
from ..modules.autoencoder import AutoEncoder
from ..modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from ..modules.model_edit import Step1XParams, Step1XEdit


class Step1XEditModelBundle:
    """Bundle containing all components of the Step1X-Edit model."""

    def __init__(self, ae=None, dit=None, llm_encoder=None, device="cuda", dtype=torch.bfloat16):
        self.ae = ae
        self.dit = dit
        self.llm_encoder = llm_encoder
        self.device = device
        self.dtype = dtype
        self.quantized = False
        self.offload = False

    def load_models(self, dit_path, ae_path, qwen2vl_model_path, max_length=256):
        print(f"Loading models from dit_path: {dit_path}, \n ae_path: {ae_path}, \n  qwen2vl_model_path: {qwen2vl_model_path}")
        """Load all model components."""
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

            # Initialize DIT model
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
            self.dit = Step1XEdit(step1x_params)

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

    def get_models(self):
        """Return all model components."""
        return {
            "ae": self.ae,
            "dit": self.dit,
            "llm_encoder": self.llm_encoder,
            "device": self.device,
            "dtype": self.dtype,
            "quantized": self.quantized,
            "offload": self.offload
        }