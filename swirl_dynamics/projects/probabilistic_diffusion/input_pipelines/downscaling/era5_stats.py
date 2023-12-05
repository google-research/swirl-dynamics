# Copyright 2023 The swirl_dynamics Authors.
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

r"""Calculate stats for an ERA5 zarr dataset.


"""

from collections.abc import Iterable
from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam

INPUT_PATH = flags.DEFINE_string("input_path", None, help="Input Zarr path")
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path")
VARIABLES = flags.DEFINE_list(
    "variables", None, help="Comma-separated variables to compute stats for"
)
START_YEAR = flags.DEFINE_string("start_year", None, "Start of time range")
END_YEAR = flags.DEFINE_string("end_year", None, "End of time range")
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
DIMS_ORDER = flags.DEFINE_list("dims_order", None, help="Order of dimensions.")

Dataset = TypeVar("Dataset", xarray.Dataset, xarray.DataArray)


def _combine_stats(mean: Dataset, var: Dataset, count: Dataset) -> Dataset:
  stats_coords = xarray.DataArray(
      ["mean", "var", "count"], dims="stats", name="stats"
  )
  return xarray.concat([mean, var, count], dim=stats_coords)


def initialize_stats(
    key: xbeam.Key, dataset: Dataset, skipna: bool = True
) -> tuple[xbeam.Key, Dataset]:
  """Initializes the statistics along a new axis."""
  mean = dataset.fillna(0) if skipna else dataset
  var = xarray.zeros_like(mean)
  count = mean.notnull() if skipna else xarray.ones_like(mean)
  new_dataset = _combine_stats(mean, var, count)
  return key.with_offsets(stats=0), new_dataset


def rekey_to_date(
    key: xbeam.Key, dataset: Dataset
) -> tuple[xbeam.Key, Dataset]:
  """Rekeys from time to year, month and day."""
  year = dataset.time.dt.year.item()
  month = dataset.time.dt.month.item()
  day = dataset.time.dt.day.item()
  start_year = int(START_YEAR.value) if START_YEAR.value else 1959
  new_key = key.with_offsets(
      time=None, year=year - start_year, month=month - 1, day=day - 1
  )
  new_dataset = dataset.squeeze("time", drop=True).expand_dims(
      year=[year], month=[month], day=[day]
  )
  return new_key, new_dataset


def drop_key(
    key: xbeam.Key, dataset: Dataset, drop: str | None = None
) -> tuple[xbeam.Key, Dataset]:
  """Drops a selected key."""
  if drop is not None:
    key_kwargs = {drop: None}
    key = key.with_offsets(**key_kwargs)
    dataset = dataset.squeeze([drop], drop=True)
  return key, dataset


class MeanAndVariance(beam.CombineFn):
  """Computes the mean and standard deviation of an xarray dataset."""

  def _split_stats(self, ds: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    return (ds.sel(stats="mean"), ds.sel(stats="var"), ds.sel(stats="count"))

  def create_accumulator(self) -> xarray.DataArray:
    return xarray.DataArray(
        data=[0, 0, 0], dims="stats", coords={"stats": ["mean", "var", "count"]}
    )

  def add_input(
      self, accumulator: Dataset, element: xarray.Dataset
  ) -> xarray.Dataset:
    mean, var, count = self._split_stats(accumulator)
    element_mean, element_var, element_count = self._split_stats(element)
    mean_increment = element_mean - mean
    new_count = count + element_count
    new_mean = mean + element_count / new_count * mean_increment
    new_var = (
        var * count / new_count
        + element_var * element_count / new_count
        + mean_increment**2 * count * element_count / new_count**2
    )
    new_accumulator = _combine_stats(new_mean, new_var, new_count)
    return new_accumulator.compute()  # pytype: disable=bad-return-type

  def merge_accumulators(
      self, accumulators: Iterable[xarray.Dataset | xarray.DataArray]
  ) -> xarray.Dataset | xarray.DataArray:
    means, variances, counts = zip(*list(map(self._split_stats, accumulators)))
    count = sum(counts)
    mean = sum([m * c / count for m, c in zip(means, counts)])
    variance = sum([
        (v + (m - mean) ** 2) * c / count
        for m, v, c in zip(means, variances, counts)
    ])
    new_accumulator = _combine_stats(mean, variance, count)
    return new_accumulator.compute()

  def extract_output(self, accumulator: xarray.Dataset) -> xarray.Dataset:
    return accumulator.compute()


def compute_final_stats(
    key: xbeam.Key, dataset: xarray.Dataset
) -> tuple[xbeam.Key, xarray.Dataset]:
  """Computes the final mean and std stats."""
  std = dataset.sel(stats="var") ** 0.5
  stats_coords = xarray.DataArray(["mean", "std"], dims="stats", name="stats")
  new_dataset = xarray.concat(
      [dataset.sel(stats="mean"), std], dim=stats_coords
  )
  if DIMS_ORDER.value is not None:
    new_dataset = new_dataset.transpose(*DIMS_ORDER.value)
  return key, new_dataset


def main(argv):
  input_store = INPUT_PATH.value
  source_dataset, source_chunks = xbeam.open_zarr(input_store)

  # Enforce year range.
  start = np.datetime64(START_YEAR.value, "D") if START_YEAR.value else None
  end = np.datetime64(END_YEAR.value, "D") if END_YEAR.value else None
  source_dataset = source_dataset.sel(time=slice(start, end))

  # Keeps only the variables requested.
  variables = VARIABLES.value + list(source_dataset.coords.keys())
  source_dataset = source_dataset.drop_vars(
      set(source_dataset) - set(variables)
  )

  # This lazy "template" allows us to setup the Zarr outputs before running the
  # pipeline. We don't really need to supply a template here because the outputs
  # are small (the template argument in ChunksToZarr is optional), but it makes
  # the pipeline slightly more efficient.
  template_ds = source_dataset.isel(time=0, drop=True).expand_dims(
      stats=["mean", "std"], axis=0
  )
  if DIMS_ORDER.value is not None:
    template_ds = template_ds.transpose(*DIMS_ORDER.value)
  template = xbeam.make_template(template_ds)
  output_chunks = {k: v for k, v in source_chunks.items() if k != "time"}
  output_chunks["stats"] = 1  # chunk size, not number of chunks

  output_store = OUTPUT_PATH.value

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({"time": 1})
        | beam.MapTuple(initialize_stats)
        | beam.MapTuple(rekey_to_date)
        | "drop day" >> beam.MapTuple(drop_key, "day")
        | "month aggregate" >> beam.CombinePerKey(MeanAndVariance())
        | "drop month" >> beam.MapTuple(drop_key, "month")
        | "year aggregate" >> beam.CombinePerKey(MeanAndVariance())
        | "drop year" >> beam.MapTuple(drop_key, "year")
        | "final aggregate" >> beam.CombinePerKey(MeanAndVariance())
        | beam.MapTuple(compute_final_stats)
        | xbeam.ConsolidateChunks(output_chunks)
        | xbeam.ChunksToZarr(output_store, template, output_chunks)
    )


if __name__ == "__main__":
  app.run(main)
