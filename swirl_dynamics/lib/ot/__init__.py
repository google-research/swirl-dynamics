# Copyright 2026 The swirl_dynamics Authors.
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

"""Optimal transport library."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from swirl_dynamics.lib.ot.sinkhorn import (
    SinkhornOutput,
    SinkhornSolver,
)

__all__ = [
    "SinkhornOutput",
    "SinkhornSolver",
]
