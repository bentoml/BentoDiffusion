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
Configuration utilities for SDConfig. All model configuration will inherit from ``onediffusion.SDConfig``.

Highlight feature: Each fields in ``onediffusion.SDConfig`` will also automatically generate a environment
variable based on its name field.

For example, the following config class:

```python
class StableDiffusionConfig(onediffusion.SDConfig):
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
```

which generates the environment ONEDIFFUSION_FLAN_T5_GENERATION_TEMPERATURE for users to configure temperature
dynamically during serve, ahead-of-serve or per requests.

Refer to ``onediffusion.SDConfig`` docstring for more information.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
import typing as t
from operator import itemgetter

import attr
import click_option_group as cog
import inflection
import orjson
from cattr.gen import make_dict_unstructure_fn
from cattr.gen import override
from deepmerge.merger import Merger

import onediffusion

from .exceptions import ForbiddenAttributeError
from .utils import DEBUG
from .utils import ENV_VARS_TRUE_VALUES
from .utils import LazyType
from .utils import bentoml_cattr
from .utils import codegen
from .utils import dantic
from .utils import first_not_none
from .utils import lenient_issubclass
from .utils import non_intrusive_setattr


if hasattr(t, "Required"):
    from typing import Required
else:
    from typing_extensions import Required

if hasattr(t, "NotRequired"):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

if hasattr(t, "dataclass_transform"):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

_T = t.TypeVar("_T")


if t.TYPE_CHECKING:
    from attr import _CountingAttr  # type: ignore
    from attr import _make_init  # type: ignore
    from attr import _make_repr  # type: ignore
    from attr import _transform_attrs  # type: ignore

    from ._types import ClickFunctionWrapper
    from ._types import F
    from ._types import O_co
    from ._types import P

    ReprArgs: t.TypeAlias = t.Iterable[tuple[str | None, t.Any]]

    DictStrAny = dict[str, t.Any]
    ListStr = list[str]
    ItemgetterAny = itemgetter[t.Any]
    FieldTransformers = t.Callable[[_T, list[attr.Attribute[t.Any]]], list[attr.Attribute[t.Any]]]
else:
    Constraint = t.Any
    ListStr = list
    DictStrAny = dict
    ItemgetterAny = itemgetter
    # NOTE: Using internal API from attr here, since we are actually
    # allowing subclass of onediffusion.SDConfig to become 'attrs'-ish
    from attr._make import _CountingAttr
    from attr._make import _make_init
    from attr._make import _make_repr
    from attr._make import _transform_attrs


__all__ = ["SDConfig"]

logger = logging.getLogger(__name__)

config_merger = Merger(
    # merge dicts
    type_strategies=[(DictStrAny, "merge")],
    # override all other types
    fallback_strategies=["override"],
    # override conflicting types
    type_conflict_strategies=["override"],
)


def _field_env_key(model_name: str, key: str, suffix: str | t.Literal[""] | None = None) -> str:
    return "_".join(filter(None, map(str.upper, ["ONEDIFFUSION", model_name, suffix.strip("_") if suffix else "", key])))


# cached it here to save one lookup per assignment
_object_getattribute = object.__getattribute__


class ModelSettings(t.TypedDict, total=False):
    """ModelSettings serve only for typing purposes as this is transcribed into SDConfig.__config__.
    Note that all fields from this dictionary will then be converted to __sdserver_*__ fields in SDConfig.
    """

    # NOTE: These required fields should be at the top, as it will be kw_only
    default_id: Required[str]
    model_ids: Required[ListStr]
    default_pipeline: Required[str]
    pipelines: Required[ListStr]

    # meta
    url: str
    requires_gpu: bool
    service_name: NotRequired[str]
    requirements: t.Optional[ListStr]

    # naming convention, only name_type is needed to infer from the class
    # as the three below it can be determined automatically
    name_type: t.Literal["dasherize", "lowercase"]
    model_name: NotRequired[str]
    start_name: NotRequired[str]
    env: NotRequired[onediffusion.utils.ModelEnv]

    # serving configuration
    timeout: int
    workers_per_resource: t.Union[int, float]


def _settings_field_transformer(
    _: type[attr.AttrsInstance], __: list[attr.Attribute[t.Any]]
) -> list[attr.Attribute[t.Any]]:
    return [
        attr.Attribute.from_counting_attr(
            k,
            dantic.Field(
                kw_only=False if t.get_origin(ann) is not Required else True,
                auto_default=True,
                use_default_converter=False,
                type=ann,
                metadata={"target": f"__sdserver_{k}__"},
                description=f"ModelSettings field for {k}.",
            ),
        )
        for k, ann in t.get_type_hints(ModelSettings).items()
    ]


@attr.define(slots=True, field_transformer=_settings_field_transformer, frozen=False)
class _ModelSettingsAttr:
    """Internal attrs representation of ModelSettings."""

    def __getitem__(self, key: str) -> t.Any:
        if key in codegen.get_annotations(ModelSettings):
            return _object_getattribute(self, key)
        raise KeyError(key)

    @classmethod
    def default(cls) -> _ModelSettingsAttr:
        _ = ModelSettings(
            default_id="__default__",
            model_ids=["__default__"],
            default_pipeline="__default__",
            pipelines=["__default__"],
            name_type="dasherize",
            requires_gpu=False,
            url="",
            requirements=None,
            timeout=int(36e6),
            service_name="",
            workers_per_resource=1,
        )
        return cls(**t.cast(DictStrAny, _))


def structure_settings(cl_: type[SDConfig], cls: type[_ModelSettingsAttr]):
    if not lenient_issubclass(cl_, SDConfig):
        raise RuntimeError(f"Given '{cl_}' must be a subclass type of 'SDConfig', got '{cl_}' instead.")

    if not hasattr(cl_, "__config__") or getattr(cl_, "__config__") is None:
        raise RuntimeError("Given SDConfig must have '__config__' that is not None defined.")

    assert cl_.__config__ is not None

    _cl_name = cl_.__name__.replace("Config", "")

    _settings_attr = _ModelSettingsAttr.default()
    try:
        cls(**t.cast(DictStrAny, cl_.__config__))
        _settings_attr = attr.evolve(_settings_attr, **t.cast(DictStrAny, cl_.__config__))
    except TypeError as e:
        raise ValueError("Either 'default_id' or 'model_ids' are emptied under '__config__' (required fields).")

    _final_value_dct: DictStrAny = {
        "model_name": inflection.underscore(_cl_name)
        if _settings_attr["name_type"] == "dasherize"
        else _cl_name.lower()
    }
    _final_value_dct["start_name"] = (
        inflection.dasherize(_final_value_dct["model_name"])
        if _settings_attr["name_type"] == "dasherize"
        else _final_value_dct["model_name"]
    )
    env = onediffusion.utils.ModelEnv(_final_value_dct["model_name"])
    _final_value_dct["env"] = env

    _final_value_dct["service_name"] = f"generated_{_final_value_dct['model_name']}_service.py"

    return attr.evolve(_settings_attr, **_final_value_dct)


bentoml_cattr.register_structure_hook(_ModelSettingsAttr, structure_settings)


def _make_env_transformer(
    cls: type[SDConfig],
    model_name: str,
    suffix: t.LiteralString | None = None,
    default_callback: t.Callable[[str, t.Any], t.Any] | None = None,
    globs: DictStrAny | None = None,
):
    def identity(_: str, x_value: t.Any) -> t.Any:
        return x_value

    default_callback = identity if default_callback is None else default_callback

    globs = {} if globs is None else globs
    globs.update(
        {
            "functools": functools,
            "__populate_env": dantic.env_converter,
            "__default_callback": default_callback,
            "__field_env": _field_env_key,
            "__suffix": suffix or "",
            "__model_name": model_name,
        }
    )

    lines: ListStr = [
        "__env = lambda field_name: __field_env(__model_name, field_name, __suffix)",
        "return [",
        "    f.evolve(",
        "        default=__populate_env(__default_callback(f.name, f.default), __env(f.name)),",
        "        metadata={",
        "            'env': f.metadata.get('env', __env(f.name)),",
        "            'description': f.metadata.get('description', '(not provided)'),",
        "        },",
        "    )",
        "    for f in fields",
        "]",
    ]
    fields_ann = "list[attr.Attribute[t.Any]]"

    return codegen.generate_function(
        cls,
        "__auto_env",
        lines,
        args=("_", "fields"),
        globs=globs,
        annotations={"_": "type[SDConfig]", "fields": fields_ann, "return": fields_ann},
    )


def _setattr_class(attr_name: str, value_var: t.Any, add_dunder: bool = False):
    """
    Use the builtin setattr to set *attr_name* to *value_var*.
    We can't use the cached object.__setattr__ since we are setting
    attributes to a class.

    If add_dunder to True, the generated globs should include a __add_dunder
    value that will be used to add the dunder methods to the class for given
    value_var
    """
    val = f"__add_dunder(cls, {value_var})" if add_dunder else value_var
    return f"setattr(cls, '{attr_name}', {val})"


_dunder_add = {}


def _make_assignment_script(
    cls: type[SDConfig], attributes: attr.AttrsInstance, _prefix: t.LiteralString = "sdserver"
) -> t.Callable[..., None]:
    """Generate the assignment script with prefix attributes __sdserver_<value>__"""
    args: ListStr = []
    globs: DictStrAny = {
        "cls": cls,
        "_cached_attribute": attributes,
        "_cached_getattribute_get": _object_getattribute.__get__,
        "__add_dunder": codegen.add_method_dunders,
    }
    annotations: DictStrAny = {"return": None}

    lines: ListStr = ["_getattr = _cached_getattribute_get(_cached_attribute)"]
    for attr_name, field in attr.fields_dict(attributes.__class__).items():
        arg_name = field.metadata.get("target", f"__{_prefix}_{inflection.underscore(attr_name)}__")
        args.append(f"{attr_name}=getattr(_cached_attribute, '{attr_name}')")
        lines.append(_setattr_class(arg_name, attr_name, add_dunder=attr_name in _dunder_add))
        annotations[attr_name] = field.type

    return codegen.generate_function(
        cls, "__assign_attr", lines, args=("cls", *args), globs=globs, annotations=annotations
    )


_reserved_namespace = {"__config__"}


@dataclass_transform(order_default=True, field_specifiers=(attr.field, dantic.Field))
def __sd_config_transform__(cls: type[SDConfig]) -> type[SDConfig]:
    kwargs: dict[str, t.Any] = {}
    non_intrusive_setattr(
        cls,
        "__dataclass_transform__",
        {
            "order_default": True,
            "field_specifiers": (attr.field, dantic.Field),
            "kwargs": kwargs,
        },
    )
    return cls


@attr.define(slots=True)
class SDConfig:
    """
    ``onediffusion.SDConfig`` is somewhat a hybrid combination between the performance of `attrs` with the
    easy-to-use interface that pydantic offer. It lives in between where it allows users to quickly formulate
    a SDConfig for any diffusion model without worrying too much about performance. It does a few things:

    - Automatic environment conversion: Each fields will automatically be provisioned with an environment
        variable, make it easy to work with ahead-of-time or during serving time
    - Familiar API: It is compatible with cattrs as well as providing a few Pydantic-2 like API,
        i.e: ``model_construct_env``, ``to_generation_config``, ``to_click_options``
    - Automatic CLI generation: It can identify each fields and convert it to compatible Click options.
        This means developers can use any of the SDConfig to create CLI with compatible-Python
        CLI library (click, typer, ...)

    > Internally, SDConfig is an attrs class. All subclass of SDConfig contains "attrs-like" features,
    > which means SDConfig will actually generate subclass to have attrs-compatible API, so that the subclass
    > can be written as any normal Python class.

    By default, all SDConfig must provide a __config__ with 'default_id' and 'model_ids'.

    All other fields are optional, and will be use default value if not set.

    ```python
    class StableDiffusionConfig(onediffusion.SDConfig):

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
    ```
    """

    Field = dantic.Field
    """Field is a alias to the internal dantic utilities to easily create
    attrs.fields with pydantic-compatible interface. For example:

    ```python
    class MyModelConfig(onediffusion.SDConfig):

        field1 = onediffusion.SDConfig.Field(...)
    ```
    """

    # NOTE: The following is handled via __init_subclass__, and is only used for TYPE_CHECKING
    if t.TYPE_CHECKING:
        # NOTE: public attributes to override
        __config__: ModelSettings | None = Field(None)
        """Internal configuration for this diffusion model. Each of the field in here will be populated
        and prefixed with __sdserver_<value>__"""

        """Users can override this subclass of any given SDConfig to provide GenerationConfig
        default value. For example:

        ```python
        class MyAwesomeModelConfig(onediffusion.SDConfig):
            class GenerationConfig:
                max_new_tokens: int = 200
                top_k: int = 10
                num_return_sequences: int = 1
                eos_token_id: int = 11
        ```
        """

        # NOTE: Internal attributes that should only be used by Onediffusion. Users usually shouldn't
        # concern any of these. These are here for pyright not to complain.
        def __attrs_init__(self, **attrs: t.Any):
            """Generated __attrs_init__ for SDConfig subclass that follows the attrs contract."""

        __attrs_attrs__: tuple[attr.Attribute[t.Any], ...] = Field(None, init=False)
        """Since we are writing our own __init_subclass__, which is an alternative way for __prepare__,
        we want onediffusion.SDConfig to be attrs-like dataclass that has pydantic-like interface.
        __attrs_attrs__ will be handled dynamically by __init_subclass__.
        """

        __sdserver_hints__: DictStrAny = Field(None, init=False)
        """An internal cache of resolved types for this SDConfig."""

        __sdserver_accepted_keys__: set[str] = Field(None, init=False)
        """The accepted keys for this SDConfig."""

        __sdserver_extras__: DictStrAny = Field(None, init=False)
        """Extra metadata for this SDConfig."""

        # NOTE: The following will be populated from __config__ and also
        # considered to be public API.
        __sdserver_default_id__: str = Field(None)
        """Return the default model to use when using 'onediffusion start <model_id>'.
        This could be one of the keys in 'self.model_ids' or custom users model.

        This field is required when defining under '__config__'.
        """

        __sdserver_model_ids__: ListStr = Field(None)
        """A list of supported pretrained models tag for this given runnable.

        For example:
            For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                                "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

        This field is required when defining under '__config__'.
        """

        __sdserver_default_pipeline__: str = Field(None)
        """Return the default pipeline to use when using 'onediffusion start <model_id>'.

        This field is required when defining under '__config__'.
        """

        __sdserver_pipelines__: ListStr = Field(None)
        """A list of supported pipelines for this given runnable.
        """

        __sdserver_url__: str = Field(None, init=False)
        """The resolved url for this SDConfig."""

        __sdserver_requires_gpu__: bool = Field(None, init=False)
        """Determines if this model is only available on GPU. By default it supports GPU and fallback to CPU."""

        __sdserver_service_name__: str = Field(None)
        """Generated service name for this SDConfig. By default, it is 'generated_{model_name}_service.py'"""

        __sdserver_requirements__: ListStr | None = Field(None)
        """The default PyPI requirements needed to run this given SD. By default, we will depend on
        bentoml, torch, transformers, diffusers, safetensors, accelerate, triton, pydantic."""

        __sdserver_name_type__: t.Literal["dasherize", "lowercase"] = Field("dasherize")
        """the default name typed for this model. "dasherize" will convert the name to lowercase and
        replace spaces with dashes. "lowercase" will convert the name to lowercase."""

        __sdserver_model_name__: str = Field(None)
        """The normalized version of __sdserver_start_name__, determined by __sdserver_name_type__"""

        __sdserver_start_name__: str = Field(None)
        """Default name to be used with `onediffusion start`"""

        __sdserver_env__: onediffusion.utils.ModelEnv = Field(None, init=False)
        """A ModelEnv instance for this SDConfig."""

        __sdserver_timeout__: int = Field(36e6)
        """The default timeout to be set for this given diffusion model."""

        __sdserver_workers_per_resource__: int | float = Field(1)
        """The number of workers per resource. This is used to determine the number of workers to use for this model.
        For example, if this is set to 0.5, then Sdserver will use 1 worker per 2 resources. If this is set to 1, then
        Sdserver will use 1 worker per resource. If this is set to 2, then Sdserver will use 2 workers per resource.

        See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy for more details.

        By default, it is set to 1.
        """



    def __init_subclass__(cls):
        """The purpose of this __init_subclass__ is that we want all subclass of SDConfig
        to adhere to the attrs contract, and have pydantic-like interface. This means we will
        construct all fields and metadata and hack into how attrs use some of the 'magic' construction
        to generate the fields.

        It also does a few more extra features: It also generate all __sdserver_*__ config from
        ModelSettings (derived from __config__) to the class.
        """
        if not cls.__name__.endswith("Config"):
            logger.warning("SDConfig subclass should end with 'Config'. Updating to %sConfig", cls.__name__)
            cls.__name__ = f"{cls.__name__}Config"

        # NOTE: auto assignment attributes generated from __config__
        _make_assignment_script(cls, bentoml_cattr.structure(cls, _ModelSettingsAttr))(cls)
        # process a fields under cls.__dict__ and auto convert them with dantic.Field
        cd = cls.__dict__
        anns = codegen.get_annotations(cls)

        # _CountingAttr is the underlying representation of attr.field
        ca_names = {name for name, attr in cd.items() if isinstance(attr, _CountingAttr)}
        these: dict[str, _CountingAttr[t.Any]] = {}
        annotated_names: set[str] = set()
        for attr_name, typ in anns.items():
            if codegen.is_class_var(typ):
                continue
            annotated_names.add(attr_name)
            val = cd.get(attr_name, attr.NOTHING)
            if not LazyType["_CountingAttr[t.Any]"](_CountingAttr).isinstance(val):
                if val is attr.NOTHING:
                    val = cls.Field(env=_field_env_key(cls.__sdserver_model_name__, attr_name))
                else:
                    val = cls.Field(default=val, env=_field_env_key(cls.__sdserver_model_name__, attr_name))
            these[attr_name] = val
        unannotated = ca_names - annotated_names
        if len(unannotated) > 0:
            missing_annotated = sorted(unannotated, key=lambda n: t.cast("_CountingAttr[t.Any]", cd.get(n)).counter)
            raise onediffusion.exceptions.MissingAnnotationAttributeError(
                f"The following field doesn't have a type annotation: {missing_annotated}"
            )

        cls.__sdserver_accepted_keys__ = set(these.keys())

        # Generate the base __attrs_attrs__ transformation here.
        attrs, base_attrs, base_attr_map = _transform_attrs(
            cls,  # the current class
            these,  # the parsed attributes we previous did
            False,  # disable auto_attribs, since we already handle these
            False,  # disable kw_only
            True,  # collect_by_mro
            field_transformer=_make_env_transformer(cls, cls.__sdserver_model_name__),
        )
        _weakref_slot = True  # slots = True
        _base_names = {a.name for a in base_attrs}
        _attr_names = tuple(a.name for a in attrs)
        _has_pre_init = bool(getattr(cls, "__attrs_pre_init__", False))
        _has_post_init = bool(getattr(cls, "__attrs_post_init__", False))
        # the protocol for attrs-decorated class
        cls.__attrs_attrs__ = attrs
        # generate a __attrs_init__ for the subclass, since we will
        # implement a custom __init__
        cls.__attrs_init__ = codegen.add_method_dunders(
            cls,
            _make_init(
                cls,  # cls (the attrs-decorated class)
                attrs,  # tuple of attr.Attribute of cls
                _has_pre_init,  # pre_init
                _has_post_init,  # post_init
                False,  # frozen
                True,  # slots
                False,  # cache_hash
                base_attr_map,  # base_attr_map
                False,  # is_exc (check if it is exception)
                None,  # cls_on_setattr (essentially attr.setters)
                attrs_init=True,  # whether to create __attrs_init__ instead of __init__
            ),
        )
        # __repr__ function with the updated fields.
        cls.__repr__ = codegen.add_method_dunders(cls, _make_repr(cls.__attrs_attrs__, None, cls))
        # Traverse the MRO to collect existing slots
        # and check for an existing __weakref__.
        existing_slots: DictStrAny = {}
        weakref_inherited = False
        for base_cls in cls.__mro__[1:-1]:
            if base_cls.__dict__.get("__weakref__", None) is not None:
                weakref_inherited = True
            existing_slots.update({name: getattr(base_cls, name) for name in getattr(base_cls, "__slots__", [])})

        if (
            _weakref_slot
            and "__weakref__" not in getattr(cls, "__slots__", ())
            and "__weakref__" not in _attr_names
            and not weakref_inherited
        ):
            _attr_names += ("__weakref__",)

        # We only add the names of attributes that aren't inherited.
        # Setting __slots__ to inherited attributes wastes memory.
        slot_names = [name for name in _attr_names if name not in _base_names]
        # There are slots for attributes from current class
        # that are defined in parent classes.
        # As their descriptors may be overridden by a child class,
        # we collect them here and update the class dict
        reused_slots = {
            slot: slot_descriptor for slot, slot_descriptor in existing_slots.items() if slot in slot_names
        }
        # __sdserver_extras__ holds additional metadata that might be usefule for users, hence we add it to slots
        slot_names = [name for name in slot_names if name not in reused_slots] + ["__sdserver_extras__"]
        cls.__slots__ = tuple(slot_names)
        # Finally, resolve the types
        if getattr(cls, "__attrs_types_resolved__", None) != cls:
            # NOTE: We will try to resolve type here, and cached it for faster use
            # It will be about 15-20ms slower comparing not resolving types.
            globs: DictStrAny = {"t": t, "typing": t, "Constraint": Constraint}
            if cls.__module__ in sys.modules:
                globs.update(sys.modules[cls.__module__].__dict__)

            cls = attr.resolve_types(cls, globalns=globs)

        # the hint cache for easier access
        cls.__sdserver_hints__ = {
            f.name: f.type for ite in map(attr.fields, (cls, )) for f in ite
        }
        cls = __sd_config_transform__(cls)

    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom SD subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    def __init__(
        self,
        *,
        __sdserver_extras__: DictStrAny | None = None,
        **attrs: t.Any,
    ):
        # create a copy of the keys as cache
        _cached_keys = tuple(attrs.keys())
        _cached_keys = tuple(k for k in _cached_keys if k in attrs)

        self.__sdserver_extras__ = first_not_none(__sdserver_extras__, default={})
        config_merger.merge(
            self.__sdserver_extras__, {k: v for k, v in attrs.items() if k not in self.__sdserver_accepted_keys__}
        )

        for k in _cached_keys:
            if k in self.__sdserver_extras__:
                del attrs[k]
        _cached_keys = tuple(k for k in _cached_keys if k in attrs)

        if DEBUG:
            logger.info(
                "Creating %s with the following attributes: %s",
                self.__class__.__name__,
                _cached_keys,
            )

        # The rest of attrs should only be the attributes to be passed to __attrs_init__
        self.__attrs_init__(**attrs)

    def __getitem__(self, item: str | t.Any) -> t.Any:
        """Allowing access SDConfig as a dictionary. The order will always evaluate as

        __sdserver_*__ > self.key > __sdserver_extras__

        This method is purely for convenience, and should not be used for performance critical code.
        """
        if not isinstance(item, str):
            raise TypeError(f"SD only supports string indexing, not {item.__class__.__name__}")
        if item in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"'{item}' is a reserved namespace for {self.__class__} and should not be access nor modified."
            )
        internal_attributes = f"__sdserver_{item}__"
        if hasattr(self, internal_attributes):
            return getattr(self, internal_attributes)
        elif hasattr(self, item):
            return getattr(self, item)
        elif item in self.__sdserver_extras__:
            return self.__sdserver_extras__[item]
        else:
            raise KeyError(item)

    def __getattribute__(self, item: str) -> t.Any:
        if item in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"'{item}' belongs to a private namespace for {self.__class__} and should not be access nor modified."
            )
        return _object_getattribute.__get__(self)(item)

    @classmethod
    def model_derivate(cls, name: str | None = None, **attrs: t.Any) -> SDConfig:
        """A helper class to generate a new SDConfig class with additional attributes.

        This is useful to modify builtin __config__ value attributes.

        ```python
        class DollyV2Config(onediffusion.SDConfig):
            ...

        my_new_class = DollyV2Config.model_derivate(default_id='...')
        ```

        Args:
            name: The name of the new class.
            **attrs: The attributes to be added to the new class. This will override
                     any existing attributes with the same name.
        """
        assert cls.__config__ is not None, "Cannot derivate a SDConfig without __config__"
        _new_cfg = {k: v for k, v in attrs.items() if k in attr.fields_dict(_ModelSettingsAttr)}
        attrs = {k: v for k, v in attrs.items() if k not in _new_cfg}
        new_cls = types.new_class(
            name or f"{cls.__name__.replace('Config', '')}DerivateConfig",
            (cls,),
            {},
            lambda ns: ns.update(
                {
                    "__config__": config_merger.merge(copy.deepcopy(cls.__dict__["__config__"]), _new_cfg),
                }
            ),
        )

        # For pickling to work, the __module__ variable needs to be set to the
        # frame where the class is created.  Bypass this step in environments where
        # sys._getframe is not defined (Jython for example) or sys._getframe is not
        # defined for arguments greater than 0 (IronPython).
        try:
            new_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass

        return new_cls(**attrs)

    def model_dump(self, flatten: bool = False, **_: t.Any):
        dumped = bentoml_cattr.unstructure(self)
        return dumped

    def model_dump_json(self, **kwargs: t.Any):
        return orjson.dumps(self.model_dump(**kwargs))

    @classmethod
    def model_construct_env(cls, **attrs: t.Any) -> t.Self:
        """A helpers that respect configuration values that
        sets from environment variables for any given configuration class.
        """
        attrs = {k: v for k, v in attrs.items() if v is not None}
        model_config = cls.__sdserver_env__.config

        env_json_string = os.environ.get(model_config, None)

        if env_json_string is not None:
            try:
                config_from_env = orjson.loads(env_json_string)
            except orjson.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse '{model_config}' as valid JSON string.") from e
        else:
            config_from_env = {}

        env_struct = bentoml_cattr.structure(config_from_env, cls)

        return attr.evolve(env_struct, **attrs)

    def model_validate_click(self, **attrs: t.Any) -> tuple[SDConfig, DictStrAny]:
        """Parse given click attributes into a SDConfig and return the remaining click attributes."""
        sd_config_attrs: DictStrAny = {}
        key_to_remove: ListStr = []

        for k, v in attrs.items():
            if k.startswith(f"{self['model_name']}_"):
                sd_config_attrs[k[len(self["model_name"] + "_") :]] = v
                key_to_remove.append(k)

        return self.model_construct_env(**sd_config_attrs), {k: v for k, v in attrs.items() if k not in key_to_remove}

    @classmethod
    @t.overload
    def to_click_options(
        cls, f: t.Callable[..., onediffusion.SDConfig]
    ) -> F[P, ClickFunctionWrapper[..., onediffusion.SDConfig]]:
        ...

    @classmethod
    @t.overload
    def to_click_options(cls, f: t.Callable[P, O_co]) -> F[P, ClickFunctionWrapper[P, O_co]]:
        ...

    @classmethod
    def to_click_options(cls, f: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """
        Convert current model to click options. This can be used as a decorator for click commands.
        Note that the identifier for all SDConfig will be prefixed with '<model_name>_*', and the generation config
        will be prefixed with '<model_name>_generation_*'.
        """

        for name, field in attr.fields_dict(cls).items():
            ty = cls.__sdserver_hints__.get(name)
            f = dantic.attrs_to_options(name, field, cls.__sdserver_model_name__, typ=ty)(f)

        return cog.optgroup.group(f"{cls.__name__} options")(f)


bentoml_cattr.register_unstructure_hook_factory(
    lambda cls: lenient_issubclass(cls, SDConfig),
    lambda cls: make_dict_unstructure_fn(cls, bentoml_cattr, _cattrs_omit_if_default=False),
)


def structure_sd_config(data: DictStrAny, cls: type[SDConfig]) -> SDConfig:
    """
    Structure a dictionary to a SDConfig object.

    Essentially, if the given dictionary contains a 'generation_config' key, then we will
    use it for SDConfig.generation_config

    Otherwise, we will filter out all keys are first in SDConfig, parse it in, then
    parse the remaining keys into SDConfig.generation_config
    """
    if not LazyType(DictStrAny).isinstance(data):
        raise RuntimeError(f"Expected a dictionary, but got {type(data)}")

    cls_attrs = {
        k: v for k, v in data.items() if k in cls.__sdserver_accepted_keys__
    }
    not_extras = list(cls_attrs)
    # The rest should be passed to extras
    data = {k: v for k, v in data.items() if k not in not_extras}

    return cls(__sdserver_extras__=data, **cls_attrs)


bentoml_cattr.register_structure_hook_func(lambda cls: lenient_issubclass(cls, SDConfig), structure_sd_config)
