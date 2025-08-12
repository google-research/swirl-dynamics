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

r"""Computes the residual between pair of high- and (interpolated) low-resolution datasets.

Example usage:

```
python third_party/py/swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_residual.par -- \
    --hires_input_path=/data/era5/selected_variables/0p25deg_hourly_7vars_windspeed_1959-2023.zarr \
    --lores_input_path=/data/era5/selected_variables/1p5deg_dailymean_7vars_windspeed_1959-2023.zarr \
    --output_path=/data/era5/selected_variables/0p25deg_minus_1p5deg_linear_hourly_7vars_windspeed_1959-2023_wrapped.zarr \
    --interp_method=linear
```


"""

from typing import TypeAlias

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam

Dataset: TypeAlias = xarray.Dataset | xarray.DataArray

# Flags.
LORES_INPUT_PATH = flags.DEFINE_string(
    "lores_input_path", None, help="Low-resolution dataset Zarr path."
)
HIRES_INPUT_PATH = flags.DEFINE_string(
    "hires_input_path", None, help="High-resolution dataset Zarr path."
)
INTERP_METHOD = flags.DEFINE_string(
    "interp_method", "nearest", help="Interpolation method."
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")


VARIABLES = [
    # Tuple elements: Hi-res var + indexer, lo-res var + indexer.
    ("2m_temperature", None, "2m_temperature", None),
    ("10m_magnitude_of_wind", None, "10m_magnitude_of_wind", None),
    ("mean_sea_level_pressure", None, "mean_sea_level_pressure", None),
    ("2m_specific_humidity", None, "specific_humidity", {"level": 1000}),
]


def subtract_dataset(
    key: xbeam.Key, datasets: dict[str, list[Dataset]]
) -> tuple[xbeam.Key, xarray.Dataset]:
  """Subtract high- and low-resolution datasets."""
  hires, lores = datasets["hires"], datasets["lores"]
  assert len(hires) == len(lores) == 1
  hires_ds, lores_ds = hires[0], lores[0]
  # Wrap longitude so that we can interpolate in the last stripe.
  lores_ds_lon0 = lores_ds.sel(longitude=[0.0]).assign_coords(
      longitude=np.asarray([360.0])
  )
  lores_ds = xarray.concat([lores_ds, lores_ds_lon0], dim="longitude")

  res = {}
  for hires_var, hires_indexer, lores_var, lores_indexer in VARIABLES:
    lores_da = lores_ds[lores_var].sel(lores_indexer, drop=True)
    hires_da = hires_ds[hires_var].sel(hires_indexer, drop=True)
    interp_coords = {
        "longitude": hires_da["longitude"],
        "latitude": hires_da["latitude"],
    }
    if "level" in hires_da.coords:
      interp_coords["level"] = hires_da["level"]
    interpolated = lores_da.interp(
        coords=interp_coords, method=INTERP_METHOD.value, assume_sorted=True
    )
    res[hires_var] = hires_da - interpolated

  res = xarray.Dataset(res)
  if "level" not in res.coords:
    key = key.with_offsets(level=None)
  return key, res


def main(argv):
  hires_store = gfile_store.GFileStore(HIRES_INPUT_PATH.value)
  hires_ds, hires_chunks = xbeam.open_zarr(hires_store, consolidated=True)

  # Select variables.
  hires_ds_sel = {}
  for hires_var, hires_indexer, *_ in VARIABLES:
    hires_ds_sel[hires_var] = hires_ds[hires_var].sel(hires_indexer, drop=True)
  hires_ds_sel = xarray.Dataset(hires_ds_sel)
  if "level" not in hires_ds_sel.coords:
    hires_chunks = {k: v for k, v in hires_chunks.items() if k != "level"}

  hires_dates = hires_ds["time"].to_numpy().astype("datetime64[D]")

  lores_store = gfile_store.GFileStore(LORES_INPUT_PATH.value)
  lores_ds = xarray.open_zarr(lores_store, chunks=None, consolidated=True)
  lores_ds = lores_ds.sel(time=hires_dates)
  lores_ds["time"] = hires_ds["time"].to_numpy()

  in_chunks = {"time": 1, "longitude": -1, "latitude": -1, "level": -1}
  out_chunks = in_chunks.copy()
  if "level" not in hires_ds_sel.coords:
    del out_chunks["level"]

  # Create template.
  template = xbeam.make_template(hires_ds_sel)

  output_store = OUTPUT_PATH.value

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    hires_pcolls = root | "HiRes to chunks" >> xbeam.DatasetToChunks(
        hires_ds, in_chunks, num_threads=16
    )

    lores_pcolls = root | "LoRes to chunks" >> xbeam.DatasetToChunks(
        lores_ds, in_chunks, num_threads=16
    )

    _ = (
        {"hires": hires_pcolls, "lores": lores_pcolls}
        | beam.CoGroupByKey()
        | beam.MapTuple(subtract_dataset)
        | xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            dim_sizes=template.sizes,
            source_chunks=out_chunks,
            target_chunks=hires_chunks,
            itemsize=4,
            max_mem=2**30 * 16,
        )
        | xbeam.ChunksToZarr(
            output_store, template, hires_chunks, num_threads=16
        )
    )


if __name__ == "__main__":
  app.run(main)
