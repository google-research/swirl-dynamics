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

r"""Calculate spatial correlations with respect to a set of locations.

The script takes as input a Zarr dataset and computes the spatial correlations
of all variables with respect to a set of locations. These spatial correlations
can then be used to obtain conditional masks for analog selection in the LOCA
downscaling scheme.

The current implementation selects locations uniformly spaced in the spatial
dimensions, however this can be extended to other location schemes easily by
modifying the `_get_locs` function. The results are stored in a Zarr dataset
with an additional `reference_point` dimension which supports unstructured
grids.

Example usage:
```
INPUT_EXPERIMENT=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc
CLIMATOLOGY_PATH=<climatology_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/compute_spatial_correlations.py \
    --input_path=${INPUT_EXPERIMENT}/hourly_d02.zarr \
    --output_path=${INPUT_EXPERIMENT}/stats/correlations_hourly_d02_2020-2030.zarr \
    --climatology_path=${CLIMATOLOGY_PATH} \
    --num_ref_points_dim1=5 \
    --num_ref_points_dim2=3 \
    --start_year=2020 \
    --end_year=2030
```

"""

from collections.abc import Iterable
import functools
import typing
from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam
from zarr.google import gfile_store

INPUT_PATH = flags.DEFINE_string("input_path", None, help="Input Zarr path")
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path")
CLIMATOLOGY_PATH = flags.DEFINE_string(
    "climatology_path",
    None,
    help="Zarr path pointing to the climatology.",
)
TIME_START = flags.DEFINE_string("start_year", None, "Start of time range")
TIME_STOP = flags.DEFINE_string("end_year", None, "End of time range")
TIME_DIM = flags.DEFINE_string(
    "time_dim", "time", help="Name for the time dimension to slice data on."
)
SPATIAL_DIMS = flags.DEFINE_list(
    "spatial_dims",
    ["south_north", "west_east"],
    help="Spatial dimensions to filter over.",
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
NUM_REF_POINTS_DIM1 = flags.DEFINE_integer(
    "num_ref_points_dim1",
    5,
    help="Number of reference points in the first spatial dimension.",
)
NUM_REF_POINTS_DIM2 = flags.DEFINE_integer(
    "num_ref_points_dim2",
    3,
    help="Number of reference points in the second spatial dimension.",
)

Dataset = TypeVar("Dataset", xr.Dataset, xr.DataArray)


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is None or TIME_STOP.value is None:
    return ds
  selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
  return ds.sel({k: v for k, v in selection.items() if k in ds.dims})


# TODO: Move this function to a common library.
def _get_climatology_mean(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological mean of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script,
  (https://github.com/google-research/weatherbench2/blob/main/scripts/compute_climatology.py)
  and statistics `mean`, and `std`. The convention is that the climatological
  means do not have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological mean of the given variables.
  """
  climatology_mean = climatology[variables]
  return typing.cast(xr.Dataset, climatology_mean.sel(**sel_kwargs).compute())


def _combine_stats(
    *,
    mean: Dataset,
    second_moment: Dataset,
    mixed_second_moment: Dataset,
    count: Dataset,
) -> Dataset:
  """Combines the statistics into a single dataset."""
  stats_coords = xr.DataArray(
      ["mean", "second_moment", "mixed_second_moment", "count"],
      dims="stats",
      name="stats",
  )
  return xr.concat(
      [mean, second_moment, mixed_second_moment, count], dim=stats_coords
  )


def compute_moments_chunk(
    key: xbeam.Key,
    dataset: Dataset,
    *,
    loc: tuple[int, int],
    clim: xr.Dataset,
    spatial_dims: tuple[str, str] = ("south_north", "west_east"),
    skipna: bool = True,
) -> tuple[xbeam.Key, Dataset]:
  """Computes the statistical moments of a single-time anomaly chunk.

  Args:
    key: The key indexing the chunk.
    dataset: The dataset chunk, with dimensions (time, *spatial_dims), where the
      time dimension has length 1.
    loc: The location of the reference point.
    clim: The climatology dataset, used to subtract the climatological mean.
    spatial_dims: The name of the two spatial dimensions of the dataset.
    skipna: Whether to skip NaN values.

  Returns:
    The key and the dataset with the statistical moments of the anomalies. The
    dataset has the same dimensions as the input dataset, with the addition of
    the `stats` dimension which indexes the statistical moments.
  """
  dataset = dataset.fillna(0) if skipna else dataset
  reference_point = {spatial_dims[0]: loc[0], spatial_dims[1]: loc[1]}
  clim_sel = dict(
      dayofyear=dataset[TIME_DIM.value].dt.dayofyear,
      hour=dataset[TIME_DIM.value].dt.hour,
      drop=True,
  )
  variables = [str(key) for key in dataset.keys()]
  clim_mean = _get_climatology_mean(clim, variables, **clim_sel)

  dataset = dataset - clim_mean
  count = dataset.notnull() if skipna else xr.ones_like(dataset)
  mean = dataset
  second_moment = dataset * dataset
  mixed_second_moment = dataset * dataset.isel(**reference_point)
  new_dataset = _combine_stats(
      mean=mean,
      second_moment=second_moment,
      mixed_second_moment=mixed_second_moment,
      count=count,
  )
  return key.with_offsets(stats=0), new_dataset.drop_vars(["hour", "dayofyear"])


def drop_key(
    key: xbeam.Key, dataset: Dataset, drop: str | None = None
) -> tuple[xbeam.Key, Dataset]:
  """Drops a selected key."""
  if drop is not None:
    key_kwargs = {drop: None}
    key = key.with_offsets(**key_kwargs)
    dataset = dataset.squeeze([drop], drop=True)
  return key, dataset


def add_reference_point_key(
    key: xbeam.Key, dataset: Dataset, *, loc: tuple[int, int], loc_offset: int
) -> tuple[xbeam.Key, Dataset]:
  """Adds a reference point key to the dataset with the correct offset."""
  key = key.with_offsets(reference_point=loc_offset)
  dataset = dataset.expand_dims(
      dim={"reference_point": (f"{loc[0]}_{loc[1]}",)}
  )
  return key, dataset


class AccumulateStatistics(beam.CombineFn):
  """Computes the first and second statistical moments of an xarray dataset."""

  def _split_stats(self, ds: Dataset) -> tuple[Dataset, ...]:
    return (
        ds.sel(stats="mean"),
        ds.sel(stats="second_moment"),
        ds.sel(stats="mixed_second_moment"),
        ds.sel(stats="count"),
    )

  def create_accumulator(self) -> xr.DataArray:
    return xr.DataArray(
        data=[0, 0, 0, 0],
        dims="stats",
        coords={
            "stats": ["mean", "second_moment", "mixed_second_moment", "count"]
        },
    )

  def add_input(self, accumulator: Dataset, element: xr.Dataset) -> xr.Dataset:
    """Updates the accumulator with information from a new element."""
    mean, second_moment, mixed_second_moment, count = self._split_stats(
        accumulator
    )
    (
        element_mean,
        element_second_moment,
        element_mixed_second_moment,
        element_count,
    ) = self._split_stats(element)
    new_count = count + element_count
    new_mean = (
        mean * count / new_count + element_mean * element_count / new_count
    )
    new_second_moment = (
        second_moment * count / new_count
        + element_second_moment * element_count / new_count
    )
    new_mixed_second_moment = (
        mixed_second_moment * count / new_count
        + element_mixed_second_moment * element_count / new_count
    )
    new_accumulator = _combine_stats(
        mean=new_mean,
        second_moment=new_second_moment,
        mixed_second_moment=new_mixed_second_moment,
        count=new_count,
    )
    return new_accumulator.compute()  # pytype: disable=bad-return-type

  def merge_accumulators(self, accumulators: Iterable[Dataset]) -> Dataset:
    means, second_moments, mixed_second_moments, counts = zip(
        *list(map(self._split_stats, accumulators))
    )
    count = sum(counts)
    mean = sum([m * c / count for m, c in zip(means, counts)])
    second_moment = sum([m * c / count for m, c in zip(second_moments, counts)])
    mixed_second_moment = sum(
        [m * c / count for m, c in zip(mixed_second_moments, counts)]
    )
    new_accumulator = _combine_stats(
        mean=mean,
        second_moment=second_moment,
        mixed_second_moment=mixed_second_moment,
        count=count,
    )
    return new_accumulator.compute()

  def extract_output(self, accumulator: xr.Dataset) -> xr.Dataset:
    return accumulator.compute()


def compute_correlation(
    key: xbeam.Key,
    dataset: xr.Dataset,
    *,
    loc: tuple[int, int],
    spatial_dims: tuple[str, str] = ("south_north", "west_east"),
) -> tuple[xbeam.Key, xr.Dataset]:
  """Computes the spatial correlation from raw statistical moments.

  Args:
    key: The key indexing the chunk.
    dataset: The dataset chunk, with dimensions (stats, *spatial_dims), where
      the `stats` dimension includes coordinates `mean`, `second_moment`, and
      `mixed_second_moment`.
    loc: The indices of the spatial location of the reference point in the input
      dataset grid.
    spatial_dims: The name of the two spatial dimensions of the dataset.

  Returns:
    The key and chunk of the correlation dataset. The output dataset has the
    same dimensions as the input dataset, without the `stats` dimension.
  """
  reference_point = {spatial_dims[0]: loc[0], spatial_dims[1]: loc[1]}
  ref_dataset = dataset.isel(**reference_point)
  cov = dataset.sel(stats="mixed_second_moment") - dataset.sel(
      stats="mean"
  ) * ref_dataset.sel(stats="mean")
  var = dataset.sel(stats="second_moment") - dataset.sel(
      stats="mean"
  ) * dataset.sel(stats="mean")
  std = xr.apply_ufunc(np.sqrt, var)
  var_reference = ref_dataset.sel(stats="second_moment") - ref_dataset.sel(
      stats="mean"
  ) * ref_dataset.sel(stats="mean")
  std_reference = xr.apply_ufunc(np.sqrt, var_reference)
  corr = cov / (std * std_reference)
  corr_key = key.with_offsets(stats=None)
  return corr_key, corr


def _get_locs(size: int, num_ref_points: int) -> np.ndarray:
  start = size / num_ref_points / 2
  locs = np.linspace(start, size - start, num_ref_points) - 1
  return np.array(locs).astype(int)


def main(argv):
  input_store = INPUT_PATH.value
  input_store = gfile_store.GFileStore(input_store)
  source_dataset, source_chunks = xbeam.open_zarr(input_store)
  source_dataset = _impose_data_selection(source_dataset)

  clim_ds = xr.open_zarr(CLIMATOLOGY_PATH.value)

  # Get the uniformly spaced locations for the reference points.
  spatial_dims = tuple(SPATIAL_DIMS.value)
  locs_dim1 = _get_locs(
      source_dataset.sizes[spatial_dims[0]], NUM_REF_POINTS_DIM1.value
  )
  locs_dim2 = _get_locs(
      source_dataset.sizes[spatial_dims[1]], NUM_REF_POINTS_DIM2.value
  )
  locs = [
      (loc_dim1, loc_dim2) for loc_dim1 in locs_dim1 for loc_dim2 in locs_dim2  # pylint: disable=g-complex-comprehension
  ]
  reference_point_coords = [f"{loc[0]}_{loc[1]}" for loc in locs]

  out_working_chunks = {k: v for k, v in source_chunks.items() if k != "time"}
  out_working_chunks["time"] = 1

  template_ds = source_dataset.isel(**{TIME_DIM.value: 0}, drop=True)
  template = xbeam.make_template(template_ds)

  transpose_dims = ("reference_point",)
  expand_dims = dict(reference_point=reference_point_coords)
  template = template.expand_dims(**expand_dims).transpose(*transpose_dims, ...)

  output_chunks = {k: v for k, v in source_chunks.items() if k != "time"}
  output_chunks["reference_point"] = 1

  output_store = OUTPUT_PATH.value
  output_store = gfile_store.GFileStore(output_store)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    pcoll = (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({TIME_DIM.value: 1})
    )

    # Branches for each analog pool.
    pcolls = []
    for loc_offset, loc in enumerate(locs):
      pcoll_tmp = (
          pcoll
          | f"{loc[0]}, {loc[1]}"
          >> beam.MapTuple(
              functools.partial(compute_moments_chunk, loc=loc, clim=clim_ds)
          )
          | f"Drop time {loc[0]}, {loc[1]}"
          >> beam.MapTuple(drop_key, TIME_DIM.value)
          | f"Add reference point key {loc[0]}, {loc[1]}"
          >> beam.MapTuple(
              functools.partial(
                  add_reference_point_key, loc=loc, loc_offset=loc_offset
              )
          )
          | f"Aggregate {loc[0]}, {loc[1]}"
          >> beam.CombinePerKey(AccumulateStatistics())
          | beam.MapTuple(functools.partial(compute_correlation, loc=loc))
      )
      pcolls.append(pcoll_tmp)

    _ = (
        pcolls
        | beam.Flatten()
        | xbeam.ChunksToZarr(output_store, template, output_chunks)
    )


if __name__ == "__main__":
  app.run(main)
