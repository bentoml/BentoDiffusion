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
[Stable Diffusion XL repository](https://github.com/Stability-AI/generative-models)
"""
from __future__ import annotations

import onediffusion


class StableDiffusionXLConfig(onediffusion.SDConfig):
    """Stable Diffusion XL is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

    Refer to [Stable Diffusion XL page](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for more information.
    """

    __config__ = {
        "timeout": 3600000,
        "url": "https://github.com/Stability-AI/generative-models",
        "default_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_ids": ["stabilityai/stable-diffusion-xl-base-1.0"],
        "default_pipeline": "text2img",
        "pipelines": ["text2img", "img2img"],
    }


START_STABLE_DIFFUSION_XL_COMMAND_DOCSTRING = """\
Run a OneDiffusion server for Stable Diffusion XL model.

\b
> See more information about Stable Diffusion XL at [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

\b
## Usage

Currently, stable-diffusion-xl only supports PyTorch. Make sure ``torch`` is available in your system.

\b
stable-diffusion-xl will use stabilityai/stable-diffusion-xl-base-1.0 as the default model. To change any to any other Stable Diffusion XL
saved pretrained, or a fine-tune model, provide ``ONEDIFFUSION_STABLE_DIFFUSION_XL_MODEL_ID='stabilityai/stable-diffusion-xl-base-1.0'``
or provide `--model-id` flag when running ``onediffusion start stable-diffusion-xl``:

\b
$ onediffusion start stable-diffusion-xl --model-id stabilityai/stable-diffusion-xl-base-1.0
"""
