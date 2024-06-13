import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REPO = "ByteDance/SDXL-Lightning"
CKPT = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

sample_prompt = "A girl smiling"

@bentoml.service(
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SDXLLightning:
    def __init__(self) -> None:
        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
            EulerDiscreteScheduler
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.unet = UNet2DConditionModel.from_config(
            BASE_MODEL_ID, subfolder="unet"
        ).to("cuda", torch.float16)

        self.unet.load_state_dict(
            load_file(hf_hub_download(REPO, CKPT),
                      device="cuda")
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL_ID,
            unet=self.unet,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )


    @bentoml.api
    def txt2img(self, prompt: str = sample_prompt) -> Image:
        # step number to match ckpt file version
        num_inference_steps = 4
        guidance_scale = 0.0
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
