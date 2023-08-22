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
import types
import typing as t
from collections import OrderedDict

import inflection

import onediffusion

from .configuration_auto import AutoConfig


if t.TYPE_CHECKING:
    from collections import _odict_items
    from collections import _odict_keys
    from collections import _odict_values

    ConfigModelOrderedDict = OrderedDict[type[onediffusion.SDConfig], type[onediffusion.SD[t.Any, t.Any]]]
    ConfigModelKeysView = _odict_keys[type[onediffusion.SDConfig], type[onediffusion.SD[t.Any, t.Any]]]
    ConfigModelValuesView = _odict_values[type[onediffusion.SDConfig], type[onediffusion.SD[t.Any, t.Any]]]
    ConfigModelItemsView = _odict_items[type[onediffusion.SDConfig], type[onediffusion.SD[t.Any, t.Any]]]
else:
    ConfigModelKeysView = ConfigModelValuesView = ConfigModelItemsView = t.Any
    ConfigModelOrderedDict = OrderedDict

logger = logging.getLogger(__name__)


class _BaseAutoSDClass:
    _model_mapping: _LazyAutoMapping

    def __init__(self, *args: t.Any, **attrs: t.Any):
        raise EnvironmentError(
            f"Cannot instantiate {self.__class__.__name__} directly. "
            "Please use '{self.__class__.__name__}.Runner(model_name)' instead."
        )

    @t.overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        pipeline: str | None = None,
        return_runner_kwargs: t.Literal[False] = ...,
        sd_config: onediffusion.SDConfig | None = ...,
        ensure_available: t.Literal[False, True] = ...,
        **attrs: t.Any,
    ) -> onediffusion.SD[t.Any, t.Any]:
        ...

    @t.overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        pipeline: str | None = None,
        return_runner_kwargs: t.Literal[True] = ...,
        sd_config: onediffusion.SDConfig | None = ...,
        ensure_available: t.Literal[False, True] = ...,
        **attrs: t.Any,
    ) -> tuple[onediffusion.SD[t.Any, t.Any], dict[str, t.Any]]:
        ...

    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        pipeline: str | None = None,
        return_runner_kwargs: bool = False,
        sd_config: onediffusion.SDConfig | None = None,
        ensure_available: bool = False,
        **attrs: t.Any,
    ) -> onediffusion.SD[t.Any, t.Any] | tuple[onediffusion.SD[t.Any, t.Any], dict[str, t.Any]]:
        """The lower level API for creating a OneDiffusion instance.

        ```python
        >>> import onediffusion
        >>> sd = onediffusion.AutoSD.for_model("stable-diffusion")
        ```
        """
        # order matters here
        runner_kwargs_name = {
            "models",
            "max_batch_size",
            "max_latency_ms",
            "method_configs",
            "scheduling_strategy",
        }
        to_runner_attrs = {k: v for k, v in attrs.items() if k in runner_kwargs_name}
        attrs = {k: v for k, v in attrs.items() if k not in to_runner_attrs}
        if cls._model_mapping.get(inflection.underscore(model_name), None, mapping_type="name2model"):
            if not isinstance(sd_config, onediffusion.SDConfig):
                # The rest of kwargs is now passed to config
                sd_config = AutoConfig.for_model(model_name, **attrs)
                attrs = sd_config.__sdserver_extras__
            # the rest of attrs will be saved to __sdserver_extras__
            sd = cls._model_mapping[type(sd_config)].from_pretrained(
                model_id,
                pipeline,
                sd_config=sd_config,
                **sd_config.__sdserver_extras__,
            )
            if ensure_available:
                logger.debug(
                    "'ensure_available=True', Downloading '%s' with 'model_id=%s' to local model store.",
                    model_name,
                    sd.model_id,
                )
                import bentoml
                from bentoml._internal.frameworks.diffusers_runners.utils import get_model_or_download
                _model_name = inflection.underscore(model_name)
                module = getattr(bentoml.diffusers_simple, _model_name)
                short_name = module.MODEL_SHORT_NAME
                get_model_or_download(short_name, sd.model_id)
            if not return_runner_kwargs:
                return sd
            return sd, to_runner_attrs
        raise ValueError(
            f"Unrecognized configuration class {sd_config.__class__} for this kind of AutoSD: {cls.__name__}.\n"
            f"SD type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )


    @classmethod
    def register(cls, config_class: type[onediffusion.SDConfig], sd_class: type[onediffusion.SD[t.Any, t.Any]]):
        """
        Register a new model for this class.

        Args:
            config_class: The configuration corresponding to the model to register.
            sd_class: The runnable to register.
        """
        if hasattr(sd_class, "config_class") and sd_class.config_class is not config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {sd_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, sd_class)


def getattribute_from_module(module: types.ModuleType, attr: t.Any) -> t.Any:
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    od_module = importlib.import_module("onediffusion")

    if module != od_module:
        try:
            return getattribute_from_module(od_module, attr)
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {od_module}!")
    else:
        raise ValueError(f"Could not find {attr} in {od_module}!")


class _LazyAutoMapping(ConfigModelOrderedDict):
    """Based on transformers.models.auto.configuration_auto._LazyAutoMapping
    This OrderedDict values() and keys() returns the list instead, so you don't
    have to do list(mapping.values()) to get the list of values.
    """

    def __init__(self, config_mapping: OrderedDict[str, str], model_mapping: OrderedDict[str, str]):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content: dict[t.Any, t.Any] = {}
        self._modules: dict[str, types.ModuleType] = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: type[onediffusion.SDConfig]) -> type[onediffusion.SD[t.Any, t.Any]]:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type: str, attr: str) -> t.Any:
        module_name = inflection.underscore(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "onediffusion.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return t.cast(ConfigModelKeysView, mapping_keys + list(self._extra_content.keys()))

    @t.overload
    def get(
        self, key: type[onediffusion.SDConfig], default: t.Any, mapping_type: t.Literal["default"] = "default"
    ) -> type[onediffusion.SD[t.Any, t.Any]]:
        ...

    @t.overload
    def get(self, key: str, default: t.Any, mapping_type: t.Literal["name2model", "name2config"] = ...) -> str:
        ...

    def get(
        self,
        key: str | type[onediffusion.SDConfig],
        default: t.Any,
        mapping_type: t.Literal["default", "name2config", "name2model"] = "default",
    ) -> str | type[onediffusion.SD[t.Any, t.Any]]:
        _supported = {"default", "name2model", "name2config"}
        if mapping_type not in _supported:
            raise RuntimeError(f"Unknown mapping type {mapping_type} (supported: {_supported})")

        if mapping_type == "default":
            if t.TYPE_CHECKING:
                # we check for lenient_issubclass below, but pyright is too dumb to understand
                assert not isinstance(key, str)
            else:
                if not onediffusion.utils.lenient_issubclass(key, onediffusion.SDConfig):
                    raise KeyError(f"Key must be a type of 'onediffusion.SDConfig', got {key} instead.")
            try:
                return self.__getitem__(key)
            except KeyError:
                return default
        else:
            mapping = self._model_mapping if mapping_type == "name2model" else self._config_mapping
            assert isinstance(key, str), f"Key must be a string type if mapping_type={mapping_type}"
            try:
                return mapping.__getitem__(key)
            except KeyError:
                return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return t.cast(ConfigModelValuesView, mapping_values + list(self._extra_content.values()))

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return t.cast(ConfigModelItemsView, mapping_items + list(self._extra_content.items()))

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item: t.Any):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key: t.Any, value: t.Any):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(f"'{key}' is already used by a OneDiffusion model.")

        self._extra_content[key] = value


__all__ = ["_BaseAutoSDClass", "_LazyAutoMapping"]
