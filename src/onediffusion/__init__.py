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
OneDiffusion
============

An open platform for operating diffusion models in production. Fine-tune, serve,
deploy, and monitor any diffusion models with ease.

* Built-in support for Stable Diffusion, ControlNet
* Option to bring your own fine-tuned diffusion models
* Online Serving with HTTP
* Native integration with BentoML
"""
from __future__ import annotations

import logging
import typing as t

from . import utils as utils
from .__about__ import __version__ as __version__
from .exceptions import MissingDependencyError


if utils.DEBUG:
    utils.set_debug_mode(True)
    utils.set_quiet_mode(False)

    utils.configure_logging()
    logging.basicConfig(level=logging.NOTSET)


_import_structure = {
    "_sd": ["SD"],
    "_configuration": ["SDConfig"],
    "_package": ["build"],
    "exceptions": [],
    "utils": [],
    "models": [],
    "cli": ["start"],
    # NOTE: models
    "models.auto": [
        "AutoConfig",
        "CONFIG_MAPPING",
        "MODEL_MAPPING_NAMES",
        # for flax support later
        # "MODEL_FLAX_MAPPING_NAMES",
    ],
    "models.stable_diffusion": ["StableDiffusionConfig"],
}

try:
    if not utils.is_torch_available():
        raise MissingDependencyError
except MissingDependencyError:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    # _import_structure["models.stable_diffusion"].extend(["StableDiffusion"])
    _import_structure["models.auto"].extend(["AutoSD", "MODEL_MAPPING"])


# declaration for OneDiffusion-related modules
if t.TYPE_CHECKING:
    from . import cli as cli
    # from . import client as client
    from . import exceptions as exceptions
    from . import models as models

    # Specific types import
    from ._configuration import SDConfig as SDConfig
    from .models.auto import CONFIG_MAPPING as CONFIG_MAPPING
    from .models.auto import AutoConfig as AutoConfig

    try:
        if not utils.is_torch_available():
            raise MissingDependencyError
    except MissingDependencyError:
        from .utils.dummy_pt_objects import *
    else:
        from .models.auto import MODEL_MAPPING as MODEL_MAPPING
        from .models.auto import AutoSD as AutoSD

else:
    import sys

    sys.modules[__name__] = utils.LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={
            "__version__": __version__,
            # The below is a special mapping that allows OneDIffusion to
            # be used as a dictionary.  This is purely for convenience
            # sake, and should not be used in performance critcal
            # code. This is also not considered as a public API.

            "__sdserver_special__": {"pt": "AutoSD"},
        },
    )
