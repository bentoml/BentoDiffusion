from __future__ import annotations

import bentoml
from PIL.Image import Image

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

sample_prompt = "A girl smiling"


@bentoml.service(
    name="bento-flux-timestep-distilled-service",
    image=bentoml.images.Image(python_version="3.11").requirements_file("requirements.txt"),
    traffic={"timeout": 300},
    envs=[{"name": "HF_TOKEN"}],
    resources={"gpu": 1, "gpu_type": "nvidia-a100-80gb"},
)
class FluxTimestepDistilled:
    @bentoml.on_startup
    def setup_pipeline(self) -> None:
        import torch
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.pipe.to("cuda")

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
