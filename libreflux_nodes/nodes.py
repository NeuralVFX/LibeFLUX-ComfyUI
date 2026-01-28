
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

class LoadLibreFluxPipeline:
    CATEGORY = "libreflux"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "neuralvfx/LibreFlux-IP-Adapter-ControlNet"}),
                "quantize": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),

            }
        }
    
    RETURN_TYPES = ("LIBRE_FLUX_PIPELINE",)
    FUNCTION = "load"
    

    def load_low_vram(self, model_path, device):
        from optimum.quanto import freeze, quantize, qint8


        dtype  = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=model_path,
            trust_remote_code=True,      
            torch_dtype=dtype,
            safety_checker=None         
        )

        # Optional way to download the weights
        hf_hub_download(repo_id=model_path,
                        filename="ip_adapter.pt",
                        local_dir=".",
                        local_dir_use_symlinks=False)

        # Load the IP Adapter First
        pipe.load_ip_adapter('ip_adapter.pt')

        # Quantize and Freeze
        quantize(
            pipe.transformer,
            weights=qint8,
            exclude=[
                "*.norm", "*.norm1", "*.norm2", "*.norm2_context",
                "proj_out", "x_embedder", "norm_out", "context_embedder",
            ],
        )

        quantize(
            pipe.ip_adapter,
            weights=qint8,
            exclude=[
                "*.norm", "*.norm1", "*.norm2", "*.norm2_context",
                "proj_out", "x_embedder", "norm_out", "context_embedder",
            ],
        )

        quantize(
            pipe.controlnet,
            weights=qint8,
            exclude=[
                "*.norm", "*.norm1", "*.norm2", "*.norm2_context",
                "proj_out", "x_embedder", "norm_out", "context_embedder",
            ],
        )

        freeze(pipe.transformer)
        freeze(pipe.ip_adapter)
        freeze(pipe.controlnet)

        return pipe


    def load_high_vram(self, model_path, device):       
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=model_path,
            trust_remote_code=True,      
            torch_dtype=dtype,
            safety_checker=None       
        )

        # Optional way to download the weights
        hf_hub_download(repo_id=model_path,
        filename="ip_adapter.pt",
        local_dir=".",
        local_dir_use_symlinks=False)

        pipe.load_ip_adapter('ip_adapter.pt')

        return pipe


    def load(self, model_path, quantize, cpu_offload ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if quantize:
            pipe = self.load_low_vram(model_path, device)
        else:
            pipe = self.load_high_vram(model_path, device)

        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return (pipe,)


class SampleLibreFlux:
    CATEGORY = "libreflux"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LIBRE_FLUX_PIPELINE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "neg_prompt": ("STRING", {"default": "", "multiline": True}),

                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "steps": ("INT", {"default": 75, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0}),
                "cfg_scale": ("FLOAT", {"default": 4.0}),
                "ip_scale": ("FLOAT", {"default": 1.0}),

            },
            "optional": {
                "control_image": ("IMAGE",),
                "ip_adapter_image": ("IMAGE",),

            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    
    def sample(self, pipeline, prompt, neg_prompt, width, height, steps, seed, cfg_scale, ip_scale, control_image=None, ip_adapter_image=None):
        # Placeholder — return a dummy black image

        # Optional way to download test Control Net Image
        """hf_hub_download(repo_id="neuralvfx/LibreFlux-IP-Adapter-ControlNet",
        filename="examples/libre_flux_control_image.png",
        local_dir=".",
        local_dir_use_symlinks=False)

        # Load Control Image
        cond = Image.open("examples/libre_flux_control_image.png").convert("RGB")
        cond = cond.resize((1024, 1024))

        # Optional way to download test IP Adapter Image
        hf_hub_download(repo_id="neuralvfx/LibreFlux-IP-Adapter-ControlNet",
        filename="examples/merc.jpeg",
        local_dir=".",
        local_dir_use_symlinks=False)

        # Load IP Adapter Image
        ip_image = Image.open("examples/merc.jpeg").convert("RGB")
        ip_image = ip_image.resize((512, 512))"""

        in_width = None
        in_height = None

        pil_control_image = None
        if control_image is not None:
            control_img_array = (control_image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_control_image = Image.fromarray(control_img_array)
        else:
            in_width = width
            in_height = height

        pil_ip_image = None
        if ip_adapter_image is not None:
            ip_img_array = (ip_adapter_image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_ip_image = Image.fromarray(ip_img_array)


        out = pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    control_image=pil_control_image, 
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    controlnet_conditioning_scale=1.0,
                    ip_adapter_image=pil_ip_image, 
                    ip_adapter_scale=ip_scale,
                    width = in_width,
                    height = in_height,
                    num_images_per_prompt=1,
                    generator= torch.Generator().manual_seed(seed),
                    return_dict=True,
                )
        images = [np.array(img).astype(np.float32) / 255.0 for img in out.images]
        images = torch.from_numpy(np.stack(images)) 
        return (images,)

NODE_CLASS_MAPPINGS = {
    "LoadLibreFluxPipeline": LoadLibreFluxPipeline,
    "SampleLibreFlux": SampleLibreFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLibreFluxPipeline": "Load LibreFlux Pipeline",
    "SampleLibreFlux": "Sample LibreFlux",
}

