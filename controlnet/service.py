from __future__ import annotations

import typing as t

import numpy as np
import PIL
from PIL.Image import Image as PIL_Image

import bentoml

CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

@bentoml.service(
    traffic={"timeout": 600},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    }
)
class ControlNet:

    def __init__(self) -> None:

        import torch
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=self.dtype,
        )

        self.vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_ID,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=self.dtype
        ).to(self.device)

    @bentoml.api
    def generate(
            self,
            image: PIL_Image,
            prompt: str,
            negative_prompt: t.Optional[str] = None,
            controlnet_conditioning_scale: t.Optional[float] = 1.0,
            num_inference_steps: t.Optional[int] = 50,
            guidance_scale: t.Optional[float] = 5.0,
    ) -> PIL_Image:
        import cv2

        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0

        if num_inference_steps is None:
            num_inference_steps = 50

        if guidance_scale is None:
            guidance_scale = 5.0

        arr = np.array(image)
        arr = cv2.Canny(arr, 100, 200)
        arr = arr[:, :, None]
        arr = np.concatenate([arr, arr, arr], axis=2)
        image = PIL.Image.fromarray(arr)
        return self.pipe(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).to_tuple()[0][0]
