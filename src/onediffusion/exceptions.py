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
Base exceptions for OneDiffusion. This extends BentoML exceptions.
"""
from __future__ import annotations

import bentoml


class OneDiffusionException(bentoml.exceptions.BentoMLException):
    """Base class for all OneDiffusion exceptions. This extends BentoMLException."""


class GpuNotAvailableError(OneDiffusionException):
    """Raised when there is no GPU available in given system."""


class ValidationError(OneDiffusionException):
    """Raised when a validation fails."""


class ForbiddenAttributeError(OneDiffusionException):
    """Raised when using an _internal field."""


class MissingAnnotationAttributeError(OneDiffusionException):
    """Raised when a field under onediffusion.SDConfig is missing annotations."""


class MissingDependencyError(BaseException):
    """Raised when a dependency is missing."""
