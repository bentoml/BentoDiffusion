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
from __future__ import annotations

import importlib
import logging
import typing as t

import bentoml
import onediffusion


if t.TYPE_CHECKING:
    import torch
else:
    torch = onediffusion.utils.LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)


class StableDiffusion(onediffusion.SD[t.Any, t.Any]):
    __sdserver_internal__ = True

    @property
    def import_kwargs(self):
        model_kwds = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        return model_kwds

    def sd_post_init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
