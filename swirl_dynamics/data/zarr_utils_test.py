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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from swirl_dynamics.data import zarr_utils
import xarray as xr


class ZarrUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("single_without_coords", False, 1),
      ("multiple_without_coords", False, 3),
      ("single_ambiguous_coords", True, 1),
      ("multiple_ambiguous_coords", True, 3),
  )
  def test_collected_metrics_use_independent_generic_dimensions(
      self, ambiguous_coords, num_metrics
  ):
    shape = (2, 3, 3)
    data = {
        f"metric_{i}": np.arange(np.prod(shape)).reshape(shape) + i
        for i in range(num_metrics)
    }
    coords = (
        xr.Dataset(
            coords={"time": range(4), "lat": range(3), "lon": range(3)}
        ).coords
        if ambiguous_coords
        else None
    )
    ds = zarr_utils.collected_metrics_to_ds(data, "time", slice(0, 2), coords)

    self.assertEqual(dict(ds.sizes), {"time": 2, "dim_0": 3, "dim_1": 3})
    self.assertEmpty(ds.coords)
    for name, expected in data.items():
      self.assertEqual(ds[name].dims, ("time", "dim_0", "dim_1"))
      np.testing.assert_array_equal(ds[name].values, expected)

  def test_collected_scalar_metrics_only_use_append_dimension(self):
    data = {"foo": np.arange(2), "bar": np.arange(2) + 10}
    ds = zarr_utils.collected_metrics_to_ds(data, "time", slice(0, 2))
    for name, values in data.items():
      self.assertEqual(ds[name].dims, ("time",))
      np.testing.assert_array_equal(ds[name].values, values)

  def test_generic_collected_metrics_can_be_written_and_appended(self):
    data = {"foo": np.arange(12).reshape(2, 3, 2), "bar": np.ones((2, 3, 2))}
    outdir = self.create_tempdir().full_path
    for start in (0, 2):
      zarr_utils.collected_metrics_to_zarr(
          data,
          out_dir=outdir,
          basename="generic_metrics",
          append_dim="time",
          append_slice=slice(start, start + 2),
          coords=None,
      )
    with xr.open_zarr(os.path.join(outdir, "generic_metrics.zarr")) as ds:
      self.assertEqual(dict(ds.sizes), {"time": 4, "dim_0": 3, "dim_1": 2})
      for name, values in data.items():
        np.testing.assert_array_equal(
            ds[name].values, np.concatenate([values, values])
        )

  def test_collected_metrics_to_ds(self):

    shape = (2, 10, 5, 3)
    data = {"foo": np.ones(shape), "bar": np.ones(shape)}
    append_dim = "time"
    append_slice = slice(2, 4)
    coord_dict = {
        append_dim: pd.date_range("2012-01-01", "2012-01-08"),
        "lon": range(shape[1]),
        "lat": range(shape[2]),
        "field": ["var1", "var2", "var3"],
    }
    coords = xr.Dataset(coords=coord_dict).coords
    ds = zarr_utils.collected_metrics_to_ds(
        data,
        append_dim,
        append_slice,
        coords,
    )
    with self.subTest("Correct output format"):
      self.assertIsInstance(ds, xr.Dataset)
      self.assertIn(append_dim, ds.dims)
      self.assertIn("field", ds.dims)
    with self.subTest("Correct coordinates"):
      self.assertEqual(ds.dims[append_dim], 2)
      self.assertSequenceAlmostEqual(
          ds.coords[append_dim], pd.date_range("2012-01-03", "2012-01-04")
      )

  def test_collected_metrics_to_zarr(self):

    shape = (2, 10, 5, 3)
    data = {"foo": np.ones(shape), "bar": np.ones(shape)}
    append_dim = "time"
    append_slice = slice(2, 4)
    coord_dict = {
        append_dim: pd.date_range("2012-01-01", "2012-01-08"),
        "lon": range(shape[1]),
        "lat": range(shape[2]),
        "field": ["var1", "var2", "var3"],
    }
    coords = xr.Dataset(coords=coord_dict).coords
    outdir = self.create_tempdir()
    zarr_utils.collected_metrics_to_zarr(
        data,
        out_dir=outdir,
        basename="test_metrics",
        append_dim=append_dim,
        append_slice=append_slice,
        coords=coords,
    )

    self.assertTrue(os.path.exists(os.path.join(outdir, "test_metrics.zarr")))

  def test_aggregated_metrics_to_ds(self):

    shape = (10, 5, 3)
    data = {"foo": np.ones(shape), "bar": np.ones(shape)}
    coord_dict = {
        "time": pd.date_range("2012-01-01", "2012-01-08"),
        "lon": range(shape[0]),
        "lat": range(shape[1]),
        "field": ["var1", "var2", "var3"],
    }
    coords = xr.Dataset(coords=coord_dict).coords
    ds = zarr_utils.aggregated_metrics_to_ds(data, coords)

    self.assertIsInstance(ds, xr.Dataset)
    self.assertIn("foo", ds)
    self.assertIn("bar", ds)
    self.assertIn("field", ds.dims)
    self.assertEqual(ds.dims["field"], 3)
    self.assertEqual(ds.dims["lon"], 10)

  def test_aggregated_metrics_to_ds_ambiguous_shape(self):

    shape = (5, 5, 3)
    data = {"foo": np.ones(shape), "bar": np.ones(shape)}
    coord_dict = {
        "time": pd.date_range("2012-01-01", "2012-01-08"),
        "lon": range(shape[0]),
        "lat": range(shape[1]),
        "field": ["var1", "var2", "var3"],
    }
    coords = xr.Dataset(coords=coord_dict).coords
    ds = zarr_utils.aggregated_metrics_to_ds(data, coords)
    # Returned coordinates are generic when ambiguous
    self.assertIsInstance(ds, xr.Dataset)
    self.assertIn("foo", ds)
    self.assertIn("bar", ds)
    self.assertIn("dim_0", ds.dims)
    self.assertEqual(ds.dims["dim_2"], 3)
    self.assertEqual(ds.dims["dim_1"], 5)

  def test_aggregated_metrics_to_zarr(self):

    shape = (10, 5, 3)
    data = {"foo": np.ones(shape), "bar": np.ones(shape)}
    coord_dict = {
        "time": pd.date_range("2012-01-01", "2012-01-08"),
        "lon": range(shape[0]),
        "lat": range(shape[1]),
        "field": ["var1", "var2", "var3"],
    }
    coords = xr.Dataset(coords=coord_dict).coords
    outdir = self.create_tempdir()
    zarr_utils.aggregated_metrics_to_zarr(
        data,
        out_dir=outdir,
        basename="test_metrics",
        coords=coords,
    )

    self.assertTrue(os.path.exists(os.path.join(outdir, "test_metrics.zarr")))

  def test_write_to_file(self):
    foo = np.ones((3,))
    outdir = self.create_tempdir()
    ds = xr.Dataset(data_vars=dict(foo=(["x"], foo)))
    zarr_utils.write_to_file(ds, outdir, "written_file")

    with self.subTest("Correct file creation"):
      self.assertTrue(os.path.exists(os.path.join(outdir, "written_file.zarr")))

    with self.subTest("Correct appending"):
      zarr_utils.write_to_file(ds, outdir, "written_file", "x")
      ds_appended = xr.open_zarr(os.path.join(outdir, "written_file.zarr"))
      self.assertEqual(ds_appended.dims["x"], 6)


if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  absltest.main()
