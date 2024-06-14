import typing as t
import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

sample_prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

@bentoml.service(
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SDXL:
    def __init__(self) -> None:
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to(device="cuda")

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            negative_prompt: t.Optional[str] = None,
            num_inference_steps: Annotated[int, Ge(1), Le(100)] = 50,
            guidance_scale: float = 7.5,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
