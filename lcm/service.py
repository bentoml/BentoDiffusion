import bentoml
from PIL.Image import Image

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

sample_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"

@bentoml.service(
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class LatentConsistency:
    def __init__(self) -> None:
        from diffusers import DiffusionPipeline, LCMScheduler
        import torch

        self.lcm_txt2img = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.lcm_txt2img.load_lora_weights(lcm_lora_id)
        self.lcm_txt2img.scheduler = LCMScheduler.from_config(self.lcm_txt2img.scheduler.config)
        self.lcm_txt2img.to(device="cuda", dtype=torch.float16)

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            num_inference_steps: int = 4,
            guidance_scale: float = 1.0,
    ) -> Image:
        image = self.lcm_txt2img(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
