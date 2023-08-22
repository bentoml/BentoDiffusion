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
    "configuration_stable_diffusion_xl": ["StableDiffusionXLConfig", "START_STABLE_DIFFUSION_XL_COMMAND_DOCSTRING",],
}

try:
    if not onediffusion.utils.is_torch_available():
        raise onediffusion.exceptions.MissingDependencyError
except onediffusion.exceptions.MissingDependencyError:
    pass
else:
    _import_structure["modeling_stable_diffusion_xl"] = ["StableDiffusionXL"]
    pass

if t.TYPE_CHECKING:
    from .configuration_stable_diffusion import START_STABLE_DIFFUSION_XL_COMMAND_DOCSTRING as START_STABLE_DIFFUSION_XL_COMMAND_DOCSTRING
    from .configuration_stable_diffusion import StableDiffusionXLConfig as StableDiffusionXLConfig

    try:
        if not onediffusion.utils.is_torch_available():
            raise onediffusion.exceptions.MissingDependencyError
    except onediffusion.exceptions.MissingDependencyError:
        pass
    else:
        pass
else:
    import sys

    sys.modules[__name__] = onediffusion.utils.LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
