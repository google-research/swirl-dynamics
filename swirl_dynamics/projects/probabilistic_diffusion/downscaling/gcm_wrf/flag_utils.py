# Copyright 2024 The swirl_dynamics Authors.
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

"""Flag utils for Xarray-Beam pipelines."""

import re
from typing import Any, Union

from absl import flags

DimValueType = Union[int, float, str]


def _chunks_string_is_valid(chunks_string: str) -> bool:
  return re.fullmatch(r'(\w+=-?\d+(,\w+=-?\d+)*)?', chunks_string) is not None


def _parse_chunks(chunks_string: str) -> dict[str, int]:
  """Parse a chunks string into a dict."""
  chunks = {}
  if chunks_string:
    for entry in chunks_string.split(','):
      key, value = entry.split('=')
      chunks[key] = int(value)
  return chunks


class _ChunksParser(flags.ArgumentParser):
  """Parser for Xarray-Beam chunks flags."""

  syntactic_help: str = (
      'comma separate list of dim=size pairs, e.g., "time=10,longitude=100"'
  )

  def parse(self, argument: str) -> dict[str, int]:
    if not _chunks_string_is_valid(argument):
      raise ValueError(f'invalid chunks string: {argument}')
    return _parse_chunks(argument)

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'dict[str, int]'


class _DimValuePairSerializer(flags.ArgumentSerializer):
  """Serializer for dim=value pairs."""

  def serialize(self, value: dict[str, int]) -> str:
    return ','.join(f'{k}={v}' for k, v in value.items())


def DEFINE_chunks(  # pylint: disable=invalid-name
    name: str,
    default: str,
    help: str,  # pylint: disable=redefined-builtin
    **kwargs: Any,
):
  """Define a flag for defining Xarray-Beam chunks."""
  parser = _ChunksParser()
  serializer = _DimValuePairSerializer()
  return flags.DEFINE(
      parser, name, default, help, serializer=serializer, **kwargs
  )
