import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

sample_prompt = "A girl smiling"

runtime_image = bentoml.images.PythonImage(
    python_version="3.11",
).requirements_file("requirements.txt")


@bentoml.service(
    name="bento-flux-timestep-distilled-service",
    image=runtime_image,
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-a100",
    },
)
class FluxTimestepDistilled:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        import torch
        from diffusers import FluxPipeline
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.pipe = FluxPipeline.from_pretrained(
            self.hf_model,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()


    @bentoml.api
    def txt2img(self, prompt: str = sample_prompt) -> Image:
        # step number to match ckpt file version
        num_inference_steps = 4
        guidance_scale = 0.0
        image = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=768,
            width=1360,
            num_inference_steps=num_inference_steps,
            max_sequence_length=256,
        ).images[0]
        return image
