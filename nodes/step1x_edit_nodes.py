import os
import torch
import numpy as np
import itertools
import math
from einops import rearrange, repeat
from torchvision.transforms import functional as F

from ..utils.model_loader import Step1XEditModelBundle
from ..utils.sampling import get_schedule
import folder_paths


class REDStep1XEditModelLoader:
    """Node for loading the Step1X-Edit model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"), {"default": "step1x-edit-i1258-FP8.safetensors"}),
                "vae": (folder_paths.get_filename_list("vae"), {"default": "ae.safetensors"}),
                "text_encoder": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "Qwen2.5-VL-7B-Instruct"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "attention_mode": (["flash-attn", "torch-sdpa", "vanilla"], {"default": "torch-sdpa"}),
                "quantized": ("BOOLEAN", {"default": True}),
                "offload": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STEP1X_MODEL_BUNDLE",)
    FUNCTION = "load_model"
    CATEGORY = "Step1X-Edit"

    def load_model(self, diffusion_model, vae, text_encoder, dtype, attention_mode, quantized, offload):
        """Load the Step1X-Edit model components."""
        dit_path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        ae_path = folder_paths.get_full_path("vae", vae)
        qwen2vl_model_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0], text_encoder)

        # Convert dtype string to torch dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model bundle
        model_bundle = Step1XEditModelBundle(device=device, dtype=torch_dtype, attention_mode=attention_mode)
        model_bundle.set_quantized(quantized)
        model_bundle.set_offload(offload)

        # Load models
        success = model_bundle.load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=640
        )

        if not success:
            raise RuntimeError("Failed to load Step1X-Edit models. Please check paths and ensure all dependencies are installed.")

        return (model_bundle,)


class REDStep1XEditGenerateNode:
    """Node for generating images using Step1X-Edit."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("STEP1X_MODEL_BUNDLE",),
                "input_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg_guidance": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "size_level": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Step1X-Edit"

    def process_diff_norm(self, diff_norm, k):
        pow_result = torch.pow(diff_norm, k)
        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def unpack(self, x, height, width):
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    def prepare(self, prompt, img, ref_image, ref_image_raw, model_bundle, device):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]

        llm_encoder = model_bundle.llm_encoder
        if model_bundle.offload:
            llm_encoder = llm_encoder.to(device)
        txt, mask = llm_encoder(prompt, ref_image_raw)
        if model_bundle.offload:
            llm_encoder = llm_encoder.cpu()
            self.cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    def cudagc(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def denoise(self, model_bundle, img, img_ids, llm_embedding, txt_ids, timesteps, cfg_guidance=4.5, mask=None, timesteps_truncate=1.0):
        device = img.device
        dit = model_bundle.dit
        if model_bundle.offload:
            dit = dit.to(device)

        for t_curr, t_prev in itertools.pairwise(timesteps):
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)

            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )
            txt, vec = dit.connector(llm_embedding, t_vec, mask)

            pred = dit(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
            )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self.process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)

            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred
            img_input_length = img.shape[1] // 2
            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )

        if model_bundle.offload:
            dit = dit.cpu()
            self.cudagc()

        return img[:, :img.shape[1] // 2]

    def input_process_image(self, img_tensor, size_level=512):
        # Convert tensor to PIL for resizing
        img = F.to_pil_image(img_tensor.squeeze(0))

        # Calculate dimensions
        w, h = img.size
        r = w / h

        if w > h:
            w_new = math.ceil(math.sqrt(size_level * size_level * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(size_level * size_level / r))
            w_new = math.ceil(h_new * r)

        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    @torch.inference_mode()
    def generate(self, model_bundle, input_image, prompt, negative_prompt, num_steps, cfg_guidance, seed, size_level):
        # ComfyUI passes images in BHWC format [batch, height, width, channels] with values in range [0, 1]
        # Convert to BCHW format for processing
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert from ComfyUI format [B, H, W, C] to [B, C, H, W]
        input_tensor = input_image.permute(0, 3, 1, 2)

        # Process the input image
        batch_size = input_tensor.shape[0]
        processed_images = []

        for i in range(batch_size):
            single_img = input_tensor[i:i+1]

            # Process image to get proper dimensions
            ref_img_pil, original_size = self.input_process_image(single_img, size_level)
            width, height = ref_img_pil.width, ref_img_pil.height

            # Convert PIL back to tensor in proper format
            ref_img_tensor = F.to_tensor(ref_img_pil).unsqueeze(0).to(device) * 2 - 1

            # Get autoencoder from model bundle
            ae = model_bundle.ae
            if model_bundle.offload:
                ae = ae.to(device)

            # Encode the image
            ref_images = ae.encode(ref_img_tensor)

            if model_bundle.offload:
                ae = ae.cpu()
                self.cudagc()

            # Setup seed
            seed_value = int(seed)
            if seed_value < 0:
                seed_value = torch.Generator(device="cpu").seed()

            # Create latent noise
            x = torch.randn(
                1,  # num_samples
                16,
                height // 8,
                width // 8,
                device=device,
                dtype=torch.bfloat16 if model_bundle.dtype == torch.bfloat16 else torch.float32,
                generator=torch.Generator(device=device).manual_seed(seed_value),
            )

            # Get timesteps
            timesteps = get_schedule(
                num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
            )

            # Cat for classifier-free guidance
            x = torch.cat([x, x], dim=0)
            ref_images = torch.cat([ref_images, ref_images], dim=0)
            ref_img_tensor = torch.cat([ref_img_tensor, ref_img_tensor], dim=0)

            # Prepare inputs
            inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_img_tensor, model_bundle=model_bundle, device=device)

            # Denoise
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x = self.denoise(
                    model_bundle=model_bundle,
                    **inputs,
                    cfg_guidance=cfg_guidance,
                    timesteps=timesteps,
                    timesteps_truncate=1.0,
                )

                # Unpack and decode
                x = self.unpack(x.float(), height, width)
                if model_bundle.offload:
                    ae = model_bundle.ae.to(device)
                x = ae.decode(x)
                if model_bundle.offload:
                    ae = ae.cpu()
                    self.cudagc()

                x = x.clamp(-1, 1)
                x = x.mul(0.5).add(0.5)

            # Convert back to PIL for resizing to original dimensions
            for img in x.float():
                pil_img = F.to_pil_image(img)
                pil_img = pil_img.resize(original_size)

                # Convert back to tensor in ComfyUI format [H, W, C]
                img_tensor = torch.tensor(np.array(pil_img)).float() / 255.0
                processed_images.append(img_tensor)

        # Stack all processed images
        output_tensor = torch.stack(processed_images)

        return (output_tensor,)