import os
import typing as t

import torch
from pydantic import BaseModel
import inflection

import bentoml
from bentoml.io import Image, JSON, Multipart
import bentoml.diffusers_simple

import onediffusion

model = os.environ.get("ONEDIFFUSION_MODEL", "{__model_name__}")  # model name
model_name = inflection.underscore(model)
model_id = os.environ.get("ONEDIFFUSION_MODEL_ID", "{__model_id__}")  # model id
pipeline = os.environ.get("ONEDIFFUSION_PIPELINE", "{__pipeline__}")  # pipeline
lora_weights = os.environ.get("ONEDIFFUSION_LORA_WEIGHTS")
lora_dir = os.environ.get("ONEDIFFUSION_LORA_DIR")

sd_config = onediffusion.AutoConfig.for_model(model)

model_name = inflection.underscore(model)
module = getattr(bentoml.diffusers_simple, model_name)
short_name = module.MODEL_SHORT_NAME

model_runner = module.create_runner(
    model_id=model_id,
    pipeline_class=pipeline,
    lora_weights=lora_weights,
    lora_dir=lora_dir,
)

input_spec_mapping = {}

class Text2ImgArgs(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str] = None
    height: t.Optional[int] = module.DEFAULT_SIZE[0]
    width: t.Optional[int] = module.DEFAULT_SIZE[1]
    num_inference_steps: t.Optional[int] = 50
    guidance_scale: t.Optional[float] = 7.5
    eta: t.Optional[float] = 0.0
    lora_weights: t.Optional[str] = None

    class Config:
        extra = "allow"

input_spec_mapping["text2img"] = JSON.from_sample(
    Text2ImgArgs(prompt="a bento box")
)

class Img2ImgArgs(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str] = None
    num_inference_steps: t.Optional[int] = 50
    guidance_scale: t.Optional[float] = 7.5
    eta: t.Optional[float] = 0.0
    lora_weights: t.Optional[str] = None
    strength: t.Optional[float] = 0.8

input_spec_mapping["img2img"] = Multipart(
    img=Image(),
    data=JSON(pydantic_model=Img2ImgArgs),
)


svc = bentoml.Service(f"onediffusion-{short_name}-service", runners=[model_runner])

endpoint_mapping = {}

def text2img(input_data):
    kwargs = input_data.dict()
    res = model_runner.run(**kwargs)
    images = res[0]
    return images[0]

endpoint_mapping["text2img"] = text2img

def img2img(img, data):
    kwargs = data.dict()
    kwargs["image"] = img
    res = model_runner.run(**kwargs)
    images = res[0]
    return images[0]

endpoint_mapping["img2img"] = img2img

svc.api(input=input_spec_mapping[pipeline], output=Image())(endpoint_mapping[pipeline])
