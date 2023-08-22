# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The following includes OneDiffusion configuration and excerpt from
[Stable Diffusion repository](https://github.com/Stability-AI/stablediffusion)
"""
from __future__ import annotations

import onediffusion


class StableDiffusionConfig(onediffusion.SDConfig):
    """Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

    Refer to [Stable Diffusion page](https://github.com/Stability-AI/stablediffusion) for more information.
    """

    __config__ = {
        "timeout": 3600000,
        "url": "https://github.com/Stability-AI/stablediffusion",
        "default_id": "stabilityai/stable-diffusion-2",
        "model_ids": [
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2",
        ],
        "default_pipeline": "text2img",
        "pipelines": ["text2img", "img2img"],
    }


START_STABLE_DIFFUSION_COMMAND_DOCSTRING = """\
Run a OneDiffusion server for stable diffusion model.

\b
> See more information about stable diffusion at [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)

\b
## Usage

Currently, stable-diffusion only supports PyTorch. Make sure ``torch`` is available in your system.

\b
Stable Diffusion Runner will use stabilityai/stable-diffusion-2 as the default model. To change any to any other stable diffusion
saved pretrained, or a fine-tuned model, provide ``ONEDIFFUSION_STABLE_DIFFUSION_MODEL_ID='runwayml/stable-diffusion-v1-5'``
or provide `--model-id` flag when running ``onediffusion start stable-diffusion``:

\b
$ onediffusion start stable-diffusion --model-id runwayml/stable-diffusion-v1-5
"""
