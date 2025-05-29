# Copyright 2025 The swirl_dynamics Authors.
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

"""Test utilities."""

import numpy as np
import pandas as pd
import xarray


def dummy_era5_dataset(
    variables=2,
    latitudes=73,
    longitudes=144,
    levels=2,
    times=365 * 4,
    freq="6H",
):
  """A mock version of the ERA5 reanalysis dataset."""
  dims = ("time", "level", "latitude", "longitude")
  shape = (times, levels, latitudes, longitudes)
  var_names = ["asn", "d2m", "e", "mn2t", "mx2t", "ptype"][:variables]
  rng = np.random.default_rng(0)
  data_vars = {
      name: (dims, rng.normal(size=shape).astype(np.float32), {"var_index": i})
      for i, name in enumerate(var_names)
  }

  latitude = np.linspace(90, 90, num=latitudes)
  longitude = np.linspace(0, 360, num=longitudes, endpoint=False)
  time = pd.date_range("1979-01-01T00", periods=times, freq=freq)
  level = np.arange(levels)
  coords = {
      "time": time,
      "level": level,
      "latitude": latitude,
      "longitude": longitude,
  }
  return xarray.Dataset(data_vars, coords, {"global_attr": "yes"})
