import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated


MODEL_ID = "stabilityai/sdxl-turbo"

sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

my_image = bentoml.images.PythonImage(python_version="3.11") \
            .requirements_file("requirements.txt")

@bentoml.service(
    image=my_image,
    traffic={"timeout": 300},
    workers=1,
    labels={'owner': 'bentoml-team', 'project': 'gallery'},
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SDXLTurbo:
    model_path = bentoml.models.HuggingFaceModel(MODEL_ID)
    
    def __init__(self) -> None:
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device="cuda")

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 1,
            guidance_scale: float = 0.0,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
