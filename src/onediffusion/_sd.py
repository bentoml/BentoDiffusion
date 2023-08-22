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

import copy
import functools
import logging
import os
import re
import subprocess
import sys
import types
import typing as t
from abc import ABC
from abc import abstractmethod

import attr
import inflection
import orjson

import bentoml
import onediffusion
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.types import ModelSignatureDict

from .exceptions import ForbiddenAttributeError
from .exceptions import GpuNotAvailableError
from .exceptions import SDServerException
from .utils import DEBUG
from .utils import LazyLoader
from .utils import ModelEnv
from .utils import bentoml_cattr
from .utils import first_not_none
from .utils import get_debug_mode
from .utils import is_torch_available
from .utils import non_intrusive_setattr
from .utils import pkg


if t.TYPE_CHECKING:
    import torch

    from bentoml._internal.runner.strategy import Strategy

    from .models.auto.factory import _BaseAutoSDClass

    class SDRunner(bentoml.Runner):
        __doc__: str
        __module__: str
        sd: onediffusion.SD[t.Any, t.Any]
        config: onediffusion.SDConfig
        sd_type: str
        identifying_params: dict[str, t.Any]

        def __call__(self, *args: t.Any, **attrs: t.Any) -> t.Any:
            ...

else:
    SDRunner = bentoml.Runner
    torch = LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)


def import_model():
    pass


def convert_diffusion_model_name(name: str | None) -> str:
    if name is None:
        raise ValueError("'name' cannot be None")
    if os.path.exists(os.path.dirname(name)):
        name = os.path.basename(name)
        logger.debug("Given name is a path, only returning the basename %s")
        return name
    return re.sub("[^a-zA-Z0-9]+", "-", name)


_reserved_namespace = {"config_class", "model", "import_kwargs"}

class SDInterface(ABC):
    """This defines the loose contract for all onediffusion.SD implementations."""

    config_class: type[onediffusion.SDConfig]
    """The config class to use for this diffusion model. If you are creating a custom SD, you must specify this class."""

    def sd_post_init(self):
        """This function can be implemented if you need to initialized any additional variables that doesn't
        concern OneDiffusion internals.
        """
        pass

    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]] | None:
        """The default import kwargs to used when importing the model.
        This will be passed into 'openllm.LLM.import_model'.
        It returns two dictionaries: one for model kwargs and one for tokenizer kwargs.
        """
        return

    def import_model(
        self, model_id: str, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any
    ) -> bentoml.Model:
        """This function can be implemented if default import_model doesn't satisfy your needs."""
        raise NotImplementedError


_M = t.TypeVar("_M")
_T = t.TypeVar("_T")


@attr.define(slots=True, repr=False)
class SD(SDInterface, t.Generic[_M, _T]):
    if t.TYPE_CHECKING:
        # The following will be populated by metaclass
        __sd_implementation__: t.Literal["pt"]
        __sd_model__: _M | None
        __sd_tag__: bentoml.Tag | None
        __sd_bentomodel__: bentoml.Model | None

        __sd_post_init__: t.Callable[[t.Self], None] | None
        __sd_custom_load__: t.Callable[[t.Self, t.Any, t.Any], None] | None
        __sd_init_kwargs__: property | None

        _model_args: tuple[t.Any, ...]
        _model_attrs: dict[str, t.Any]

        bettertransformer: bool

    def __init_subclass__(cls):
        cd = cls.__dict__
        prefix_class_name_config = cls.__name__
        implementation = "pt"

        cls.__sd_implementation__ = implementation
        config_class = onediffusion.AutoConfig.infer_class_from_name(prefix_class_name_config)

        if "__sdserver_internal__" in cd:
            if "config_class" not in cd:
                cls.config_class = config_class
            else:
                logger.debug(f"Using config class {cd['config_class']} for {cls.__name__}.")
        else:
            if "config_class" not in cd:
                raise RuntimeError(
                    "Missing required key 'config_class'. Make sure to define it within the SD subclass."
                )

        if cls.import_model is SDInterface.import_model:
            # using the default import model
            setattr(cls, "import_model", import_model)
        else:
            logger.debug("Custom 'import_model' will be used when loading model %s", cls.__name__)

        cls.__sd_post_init__ = None if cls.sd_post_init is SDInterface.sd_post_init else cls.sd_post_init
        cls.__sd_init_kwargs__ = None if cls.import_kwargs is SDInterface.import_kwargs else cls.import_kwargs

        for attr in {"bentomodel", "tag", "model"}:
            setattr(cls, f"__sd_{attr}__", None)

    # The following is the similar interface to HuggingFace pretrained protocol.
    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        pipeline: str | None = None,
        sd_config: onediffusion.SDConfig | None = None,
        *args: t.Any,
        **attrs: t.Any,
    ) -> SD[_M, _T]:
        return cls(
            model_id=model_id,
            pipeline=pipeline,
            sd_config=sd_config,
            *args,
            **attrs,
        )

    def __init__(
        self,
        model_id: str | None = None,
        pipeline: str | None = None,
        sd_config: onediffusion.SDConfig | None = None,
        *args: t.Any,
        **attrs: t.Any,
    ):
        """Initialize the SD with given pretrained model.

        Note:
        - *args to be passed to the model.
        - **attrs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
        - for tokenizer kwargs, it should be prefixed with _tokenizer_*

        For custom pretrained path, it is recommended to pass in 'onediffusion_model_version' alongside with the path
        to ensure that it won't be loaded multiple times.
        Internally, if a pretrained is given as a HuggingFace repository path , OneDiffusion will usethe commit_hash
        to generate the model version.

        For better consistency, we recommend users to also push the fine-tuned model to HuggingFace repository.

        If you need to overwrite the default ``import_model``, implement the following in your subclass:

        ```python
        def import_model(
            self,
            model_id: str,
            tag: bentoml.Tag,
            *args: t.Any,
            tokenizer_kwds: dict[str, t.Any],
            **attrs: t.Any,
        ):
            return bentoml.transformers.save_model(
                tag,
                model,
            )
        ```

        If your import model doesn't require customization, you can simply pass in `import_kwargs`
        at class level that will be then passed into The default `import_model` implementation.
        See ``bentoml.diffusers_simple.stable_diffusion`` for example.

        ```python
        stable_diffusion_runner = bentoml.diffusers_simple.create_runner(
            "stable_diffusion", torch_dtype=torch.bfloat16, device_map="gpu"
        )
        ```

        Note: If you implement your own `import_model`, then `import_kwargs` will be the
        default kwargs for every load.

        Note that this tag will be generated based on `self.default_id` or the given `pretrained` kwds.
        passed from the __init__ constructor.

        ``sd_post_init`` can also be implemented if you need to do any additional
        initialization after everything is setup.
        """

        sdserver_model_version = attrs.pop("sdserver_model_version", None)

        # # low_cpu_mem_usage is only available for model
        # # this is helpful on system with low memory to avoid OOM
        # low_cpu_mem_usage = attrs.pop("low_cpu_mem_usage", True)

        if sd_config is not None:
            logger.debug("Using provided SDConfig to initialize SD instead of from default: %r", sd_config)
            self.config = sd_config
        else:
            self.config = self.config_class.model_construct_env(**attrs)
            # The rests of the kwargs that is not used by the config class should be stored into __sdserver_extras__.
            attrs = self.config["extras"]

        model_kwds = {}
        if self.__sd_init_kwargs__:
            if t.TYPE_CHECKING:
                # the above meta value should determine that this SD has custom kwargs
                assert self.import_kwargs
            model_kwds = self.import_kwargs
            logger.debug(
                "'%s' default kwargs for model: '%s', tokenizer: '%s'",
                self.__class__.__name__,
                model_kwds,
            )

        if model_id is None:
            model_id = os.environ.get(self.config["env"].model_id, self.config["default_id"])

        if pipeline is None:
            pipeline = os.environ.get(self.config["env"].pipeline, self.config["default_pipeline"])

        # NOTE: This is the actual given path or pretrained weight for this SD.
        assert model_id is not None
        self._model_id = model_id
        self.pipeline = pipeline

        # parsing model kwargs
        model_kwds.update({k: v for k, v in attrs.items()})


        # NOTE: Save the args and kwargs for latter load
        self._model_args = args
        self._model_attrs = model_kwds
        self._sdserver_model_version = sdserver_model_version

        if self.__sd_post_init__:
            self.sd_post_init()


    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom SD subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    def __repr__(self) -> str:
        keys = {"model_id", "runner_name", "sd_type", "config"}
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in keys)})"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def runner_name(self) -> str:
        return f"onediffusion-{self.config['start_name']}-runner"

    @property
    def sd_type(self) -> str:
        return convert_diffusion_model_name(self._model_id)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {
            "configuration": self.config.model_dump_json().decode(),
            "model_ids": orjson.dumps(self.config["model_ids"]).decode(),
        }


