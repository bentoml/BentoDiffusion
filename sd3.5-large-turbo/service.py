import typing as t
import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated


MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"

sample_prompt = "A cat holding a sign that says hello world"

@bentoml.service(
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-a100",
    },
)
class SD35LargeTurbo:
    def __init__(self) -> None:
        import torch
        from diffusers import StableDiffusion3Pipeline

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to(device="cuda")

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            negative_prompt: t.Optional[str] = None,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 4,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        ).images[0]
        return image
