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

import typing as t

import onediffusion


_import_structure = {
    "configuration_stable_diffusion": ["StableDiffusionConfig", "START_STABLE_DIFFUSION_COMMAND_DOCSTRING",],
}

try:
    if not onediffusion.utils.is_torch_available():
        raise onediffusion.exceptions.MissingDependencyError
except onediffusion.exceptions.MissingDependencyError:
    pass
else:
    _import_structure["modeling_stable_diffusion"] = ["StableDiffusion"]
    pass

if t.TYPE_CHECKING:
    from .configuration_stable_diffusion import START_STABLE_DIFFUSION_COMMAND_DOCSTRING as START_STABLE_DIFFUSION_COMMAND_DOCSTRING
    from .configuration_stable_diffusion import StableDiffusionConfig as StableDiffusionConfig

    try:
        if not onediffusion.utils.is_torch_available():
            raise onediffusion.exceptions.MissingDependencyError
    except onediffusion.exceptions.MissingDependencyError:
        pass
    else:
        #from .modeling_dolly_v2 import DollyV2 as DollyV2
        pass
else:
    import sys

    sys.modules[__name__] = onediffusion.utils.LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
