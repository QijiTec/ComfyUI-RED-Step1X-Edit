import torch
import numpy as np
from typing import Optional, Dict, Any, Union


class TeaCacheForStep1XEdit:
    """
    TeaCache implementation for Step1X-Edit.
    This implementation is based on TeaCache4FLUX and adapted for Step1X-Edit.
    """
    def __init__(self,
                 rel_l1_thresh: float = 0.6,
                 verbose: bool = False):
        """
        Initialize TeaCache for Step1X-Edit.

        Args:
            rel_l1_thresh: Threshold for relative L1 distance to determine when to compute new residuals.
                           Higher values = faster inference but potential quality loss.
                           Recommended values:
                           - 0.25: ~1.5x speedup
                           - 0.4: ~1.8x speedup
                           - 0.6: ~2.0x speedup (recommended)
                           - 0.8: ~2.25x speedup
            verbose: Whether to print verbose output during inference
        """
        self.enable_teacache = True
        self.rel_l1_thresh = rel_l1_thresh
        self.verbose = verbose

        # Initialize TeaCache state
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0
        self.cnt = 0
        self.num_steps = 0

        # Rescaling coefficients from TeaCache4FLUX
        self.coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]

    def reset(self, num_steps: int):
        """Reset TeaCache state for a new denoising process."""
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0
        self.cnt = 0
        self.num_steps = num_steps

    def should_calculate(self, modulated_inp: torch.Tensor) -> bool:
        """
        Determine whether to compute a new residual or reuse the cached one.

        Args:
            modulated_inp: The modulated input tensor

        Returns:
            bool: True if we should calculate a new residual, False if we should reuse the cached one
        """
        # Always calculate for the first and last step
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # Calculate the relative L1 distance between current and previous modulated input
            rel_l1 = ((modulated_inp - self.previous_modulated_input).abs().mean() /
                      self.previous_modulated_input.abs().mean()).cpu().item()

            # Apply rescaling polynomial
            rescale_func = np.poly1d(self.coefficients)
            self.accumulated_rel_l1_distance += rescale_func(rel_l1)

            # Determine if we should calculate based on accumulated distance
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
                if self.verbose:
                    print(f"Step {self.cnt}: Reusing cached residual (accumulated_rel_l1={self.accumulated_rel_l1_distance:.4f})")
            else:
                should_calc = True
                if self.verbose:
                    print(f"Step {self.cnt}: Computing new residual (accumulated_rel_l1={self.accumulated_rel_l1_distance:.4f})")
                self.accumulated_rel_l1_distance = 0

        # Store current modulated input for next step
        self.previous_modulated_input = modulated_inp.clone()
        self.cnt += 1

        # Reset counter at the end of the denoising process
        if self.cnt == self.num_steps:
            self.cnt = 0

        return should_calc

    def store_residual(self, ori_hidden_states: torch.Tensor, hidden_states: torch.Tensor):
        """Store the residual for future reuse."""
        self.previous_residual = hidden_states - ori_hidden_states