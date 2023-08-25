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
Telemetry related for OneDiffusion tracking.

Users can disable this with ONEDIFFUSION_DO_NOT_TRACK envvar.
"""
from __future__ import annotations

import contextlib
import functools
import os
import typing as t

import attr

from bentoml._internal.utils import analytics as _internal_analytics
from bentoml._internal.utils.analytics import usage_stats as _internal_usage


if t.TYPE_CHECKING:
    import onediffusion

from ..__about__ import __version__


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

# This variable is a proxy that will control BENTOML_DO_NOT_TRACK
ONEDIFFUSION_DO_NOT_TRACK = "ONEDIFFUSION_DO_NOT_TRACK"

DO_NOT_TRACK = os.environ.get(ONEDIFFUSION_DO_NOT_TRACK, str(False)).upper()


@functools.lru_cache(maxsize=1)
def do_not_track() -> bool:
    return DO_NOT_TRACK in ENV_VARS_TRUE_VALUES


@_internal_usage.silent
def track(event_properties: _internal_analytics.schemas.EventMeta):
    if do_not_track():
        return
    _internal_analytics.track(event_properties)


@contextlib.contextmanager
def set_bentoml_tracking():
    original_value = os.environ.pop(_internal_analytics.BENTOML_DO_NOT_TRACK, str(False))
    try:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = str(do_not_track())
        yield
    finally:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = original_value


@attr.define
class OneDiffusionCliEvent(_internal_analytics.schemas.EventMeta):
    cmd_group: str
    cmd_name: str
    onediffusion_version: str = __version__

    # NOTE: reserved for the do_not_track logics
    duration_in_ms: t.Any = attr.field(default=None)
    error_type: str = attr.field(default=None)
    return_code: int = attr.field(default=None)


@attr.define
class StartInitEvent(_internal_analytics.schemas.EventMeta):
    model_name: str
    sd_config: t.Dict[str, t.Any] = attr.field(default=None)

    @staticmethod
    def handler(sd_config: onediffusion.SDConfig) -> StartInitEvent:
        return StartInitEvent(model_name=sd_config["model_name"], sd_config=sd_config.model_dump())


def track_start_init(
    sd_config: onediffusion.SDConfig,
):
    if do_not_track():
        return
    track(StartInitEvent.handler(sd_config))
