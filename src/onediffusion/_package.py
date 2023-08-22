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
Any build-related utilities. This is used for CI.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import typing as t
from pathlib import Path

import fs
import inflection

import bentoml
import onediffusion
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.frameworks.diffusers_runners.utils import get_model_or_download

from .utils import ModelEnv
from .utils import first_not_none
from .utils import is_torch_available
from .utils import pkg


if t.TYPE_CHECKING:
    from fs.base import FS

    from .models.auto.factory import _BaseAutoSDClass

logger = logging.getLogger(__name__)

ONEDIFFUSION_DEV_BUILD = "ONEDIFFUSION_DEV_BUILD"


def build_editable(path: str) -> str | None:
    """Build OneDiffusion if the ONEDIFFUSION_DEV_BUILD environment variable is set."""
    if str(os.environ.get(ONEDIFFUSION_DEV_BUILD, False)).lower() != "true":
        return

    # We need to build the package in editable mode, so that we can import it
    from build import ProjectBuilder
    from build.env import IsolatedEnvBuilder

    module_location = pkg.source_locations("onediffusion")
    if not module_location:
        raise RuntimeError(
            "Could not find the source location of OneDiffusion. Make sure to unset"
            " ONEDIFFUSION_DEV_BUILD if you are developing OneDiffusion."
        )
    pyproject_path = Path(module_location).parent.parent / "pyproject.toml"
    if os.path.isfile(pyproject_path.__fspath__()):
        logger.info("OneDiffusion is installed in editable mode. Generating built wheels...")
        with IsolatedEnvBuilder() as env:
            builder = ProjectBuilder(pyproject_path.parent)
            builder.python_executable = env.executable
            builder.scripts_dir = env.scripts_dir
            env.install(builder.build_system_requires)
            return builder.build("wheel", path, config_settings={"--global-option": "--quiet"})
    raise RuntimeError(
        "Custom OneDiffusion build is currently not supported. Please install OneDiffusion from PyPI or built it from Git source."
    )


def construct_python_options(
    sd: onediffusion.SD[t.Any, t.Any],
    sd_fs: FS,
    extra_dependencies: tuple[str, ...] | None = None,
) -> PythonOptions:
    packages = ["onediffusion"]
    # NOTE: add onediffusion to the default dependencies
    # if users has onediffusion custom built wheels, it will still respect
    # that since bentoml will always install dependencies from requirements.txt
    # first, then proceed to install everything inside the wheels/ folder.
    if extra_dependencies is not None:
        packages += [f"onediffusion[{k}]" for k in extra_dependencies]

    if sd.config["requirements"] is not None:
        packages.extend(sd.config["requirements"])

    if not (str(os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD", False)).lower() == "false"):
        packages.append(f"bentoml>={'.'.join([str(i) for i in pkg.pkg_version_info('bentoml')])}")

    env: ModelEnv = sd.config["env"]
    framework_envvar = env["framework_value"]
    assert is_torch_available(), "PyTorch is not available. Make sure to have it locally installed."
    packages.extend([f"torch>={importlib.metadata.version('torch')}"])

    wheels: list[str] = []
    built_wheels = build_editable(sd_fs.getsyspath("/"))
    if built_wheels is not None:
        wheels.append(sd_fs.getsyspath(f"/{built_wheels.split('/')[-1]}"))

    return PythonOptions(packages=packages, wheels=wheels, lock_packages=True)


def construct_docker_options(
    sd: onediffusion.SD[t.Any, t.Any],
    _: FS,
    workers_per_resource: int | float,
) -> DockerOptions:
    _bentoml_config_options = os.environ.pop("BENTOML_CONFIG_OPTIONS", "")
    _bentoml_config_options_opts = [
        "api_server.traffic.timeout=36000",  # NOTE: Currently we hardcode this value
        f'runners."sd-{sd.config["start_name"]}-runner".traffic.timeout={sd.config["timeout"]}',
        f'runners."sd-{sd.config["start_name"]}-runner".workers_per_resource={workers_per_resource}',
    ]
    _bentoml_config_options += " " if _bentoml_config_options else "" + " ".join(_bentoml_config_options_opts)
    env: ModelEnv = sd.config["env"]

    env_dict = {
        env.framework: env.framework_value,
        env.config: f"'{sd.config.model_dump_json().decode()}'",
        "ONEDIFFUSION_MODEL": sd.config["model_name"],
        "ONEDIFFUSION_MODEL_ID": sd.model_id,
        "ONEDIFFUSION_PIPELINE": sd.pipeline,
        "BENTOML_DEBUG": str(get_debug_mode()),
        "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
    }

    # NOTE: Torch 2.0 currently only support 11.6 as the latest CUDA version
    return DockerOptions(cuda_version="11.6", env=env_dict, system_packages=["git"])


@t.overload
def build(
    model_name: str,
    *,
    model_id: str | None = ...,
    pipeline: str | None = ...,
    _extra_dependencies: tuple[str, ...] | None = ...,
    _workers_per_resource: int | float | None = ...,
    _overwrite_existing_bento: bool = ...,
    __cli__: t.Literal[False] = ...,
    **attrs: t.Any,
) -> bentoml.Bento:
    ...


@t.overload
def build(
    model_name: str,
    *,
    model_id: str | None = ...,
    pipeline: str | None = ...,
    _extra_dependencies: tuple[str, ...] | None = ...,
    _workers_per_resource: int | float | None = ...,
    _overwrite_existing_bento: bool = ...,
    __cli__: t.Literal[True] = ...,
    **attrs: t.Any,
) -> tuple[bentoml.Bento, bool]:
    ...


def _build_bento(
    bento_tag: bentoml.Tag,
    service_name: str,
    sd_fs: FS,
    sd: onediffusion.SD[t.Any, t.Any],
    workers_per_resource: int | float,
    extra_dependencies: tuple[str, ...] | None = None,
) -> bentoml.Bento:
    framework_envvar = sd.config["env"]["framework_value"]
    labels = {}
    labels.update({"_type": sd.sd_type, "_framework": framework_envvar})
    logger.info("Building Bento for diffusion model '%s'", sd.config["start_name"])
    return bentoml.bentos.build(
        f"{service_name}:svc",
        name=bento_tag.name,
        labels=labels,
        description=f"OneDiffusion service for {sd.config['start_name']}",
        include=list(
            sd_fs.walk.files(filter=["*.py"])
        ),  # NOTE: By default, we are using _service.py as the default service, for now.
        exclude=["/venv", "__pycache__/", "*.py[cod]", "*$py.class"],
        python=construct_python_options(sd, sd_fs, extra_dependencies),
        docker=construct_docker_options(sd, sd_fs, workers_per_resource),
        version=bento_tag.version,
        build_ctx=sd_fs.getsyspath("/"),
    )


def build(
    model_name: str,
    *,
    model_id: str | None = None,
    pipeline: str | None = None,
    _extra_dependencies: tuple[str, ...] | None = None,
    _workers_per_resource: int | float | None = None,
    _overwrite_existing_bento: bool = False,
    __cli__: bool = False,
    **attrs: t.Any,
) -> tuple[bentoml.Bento, bool] | bentoml.Bento:
    """Package a diffusion model into a Bento.

    The diffusion model will be built into a BentoService with the following structure:

    Other parameters including model_name, model_id, pipeline and attrs will be passed to the OneDiffusion class itself.
    """

    _previously_built = False
    current_model_envvar = os.environ.pop("ONEDIFFUSION_MODEL", None)
    current_model_id_envvar = os.environ.pop("ONEDIFFUSION_MODEL_ID", None)

    sd_config = onediffusion.AutoConfig.for_model(model_name)

    logger.info("Packing '%s' into a Bento with kwargs=%s...", model_name, attrs)

    # NOTE: We set this environment variable so that our service.py logic won't raise RuntimeError
    # during build. This is a current limitation of bentoml build where we actually import the service.py into sys.path
    try:
        os.environ["ONEDIFFUSION_MODEL"] = inflection.underscore(model_name)

        framework_envvar = sd_config["env"]["framework_value"]
        sd = t.cast(
            "_BaseAutoSDClass",
            onediffusion[framework_envvar],  # type: ignore (internal API)
        ).for_model(
            model_name,
            model_id=model_id,
            pipeline=pipeline,
            sd_config=sd_config,
            **attrs,
        )

        os.environ["ONEDIFFUSION_MODEL_ID"] = sd.model_id

        labels = {}
        labels.update({"_type": sd.sd_type, "_framework": framework_envvar})
        target_service_name = f"generated_{sd_config['model_name']}_service.py"
        workers_per_resource = first_not_none(_workers_per_resource, default=sd_config["workers_per_resource"])

        with fs.open_fs(f"temp://od_{sd_config['model_name']}") as sd_fs:

            _model_name = inflection.underscore(model_name)

            # add service.py definition to this temporary folder
            base_dir = os.path.dirname(__file__)
            src_service_path = os.path.join(base_dir, "models", _model_name, "service.py")
            target_service_name = f"generated_{sd_config['model_name']}_service.py"

            with open(src_service_path) as f:
                import bentoml.diffusers_simple
                module = getattr(bentoml.diffusers_simple, _model_name)
                short_name = module.MODEL_SHORT_NAME
                _model_id = model_id or module.DEFAULT_MODEL_ID
                bento_model = get_model_or_download(
                    model_name=short_name,
                    model_id=_model_id,
                )
                model_version = bento_model.tag.version

                content = f.read()
                content = content.replace(r"{__model_name__}", model_name)
                content = content.replace(r"{__model_id__}", sd.model_id)
                content = content.replace(r"{__pipeline__}", sd.pipeline)
                sd_fs.writetext(target_service_name, content)

            bento_tag = bentoml.Tag.from_taglike(f"{sd.sd_type}-service:{model_version}")
            try:
                bento = bentoml.get(bento_tag)
                if _overwrite_existing_bento:
                    logger.info("Overwriting previously saved Bento.")
                    bentoml.delete(bento_tag)
                    bento = _build_bento(
                        bento_tag,
                        target_service_name,
                        sd_fs,
                        sd,
                        workers_per_resource=workers_per_resource,
                        extra_dependencies=_extra_dependencies,
                    )
                _previously_built = True
            except bentoml.exceptions.NotFound:
                logger.info("Building Bento for diffusion model '%s'", sd_config["start_name"])
                bento = _build_bento(
                    bento_tag,
                    target_service_name,
                    sd_fs,
                    sd,
                    workers_per_resource=workers_per_resource,
                    extra_dependencies=_extra_dependencies,
                )
            return (bento, _previously_built) if __cli__ else bento
    except Exception as e:
        logger.error("\nException caught during building diffusion model %s: \n", model_name, exc_info=e)
        raise
    finally:
        del os.environ["ONEDIFFUSION_MODEL"]
        del os.environ["ONEDIFFUSION_MODEL_ID"]
        # restore original ONEDIFFUSION_MODEL envvar if set.
        if current_model_envvar is not None:
            os.environ["ONEDIFFUSION_MODEL"] = current_model_envvar
        if current_model_id_envvar is not None:
            os.environ["ONEDIFFUSION_MODEL_ID"] = current_model_id_envvar
