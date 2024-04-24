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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from swirl_dynamics.data import zarr_utils
import xarray as xr


class ZarrUtilsTest(parameterized.TestCase):

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
