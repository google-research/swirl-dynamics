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

"""Regression tests for aggregate exports with short time coordinates."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_dynamics.data import zarr_utils
import xarray as xr


def _coords(times):
  return xr.Dataset(
      coords={
          'time': np.asarray(times, dtype='datetime64[h]'),
          'latitude': np.array([10.0, 20.0, 30.0]),
      }
  ).coords


class AggregatedMetricTimeTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'times': ['2026-01-15T12'],
          'expected': 'Aggregated metrics at 2026-01-15T12.',
      },
      {
          'times': ['2026-01-15T12', '2026-01-15T18'],
          'expected': (
              'Aggregated metrics from 2026-01-15T12 to 2026-01-15T18'
              ' with 6-hourly frequency.'
          ),
      },
  )
  def test_time_description_with_one_or_multiple_steps(self, times, expected):
    value = np.arange(len(times) * 3, dtype=np.float32).reshape(len(times), 3)
    ds = zarr_utils.aggregated_metrics_to_ds({'field': value}, _coords(times))
    self.assertEqual(ds.attrs['description'], expected)
    self.assertEqual(ds['field'].dims, ('time', 'latitude'))
    np.testing.assert_array_equal(ds['field'].values, value)
    np.testing.assert_array_equal(
        ds['time'].values, _coords(times)['time'].values
    )
    np.testing.assert_array_equal(ds['latitude'].values, [10.0, 20.0, 30.0])

  def test_single_time_coordinate_unused_by_metric(self):
    times = ['2026-01-15T12']
    ds = zarr_utils.aggregated_metrics_to_ds(
        {'field': np.array([1.0, 2.0, 3.0])}, _coords(times)
    )
    self.assertEqual(ds['field'].dims, ('latitude',))
    self.assertEqual(ds.sizes['time'], 1)
    self.assertEqual(
        ds.attrs['description'], 'Aggregated metrics at 2026-01-15T12.'
    )

  def test_empty_time_coordinate_has_generic_description(self):
    ds = zarr_utils.aggregated_metrics_to_ds(
        {'field': np.empty((0, 3), dtype=np.float32)}, _coords([])
    )
    self.assertEqual(ds['field'].dims, ('time', 'latitude'))
    self.assertEqual(ds.sizes['time'], 0)
    self.assertEqual(ds.attrs['description'], 'Aggregated metrics.')

  def test_single_timestamp_with_ambiguous_coordinate_lengths(self):
    coords = xr.Dataset(
        coords={
            'time': np.array(['2026-01-15T12'], dtype='datetime64[h]'),
            'member': [42],
            'latitude': [10.0, 20.0, 30.0],
        }
    ).coords
    ds = zarr_utils.aggregated_metrics_to_ds({'field': np.ones((1, 3))}, coords)
    self.assertEqual(ds['field'].dims, ('dim_0', 'dim_1'))
    self.assertEqual(
        ds.attrs['description'], 'Aggregated metrics at 2026-01-15T12.'
    )

  def test_single_time_zarr_round_trip(self):
    times = ['2026-01-15T12']
    out_dir = self.create_tempdir()
    zarr_utils.aggregated_metrics_to_zarr(
        {'field': np.array([[1.0, 2.0, 3.0]])},
        out_dir=out_dir.full_path,
        basename='aggregate_single_time',
        coords=_coords(times),
    )
    with xr.open_zarr(
        os.path.join(out_dir.full_path, 'aggregate_single_time.zarr')
    ) as ds:
      self.assertEqual(ds.sizes['time'], 1)
      np.testing.assert_array_equal(ds['field'].values, [[1.0, 2.0, 3.0]])
      self.assertEqual(
          ds.attrs['description'], 'Aggregated metrics at 2026-01-15T12.'
      )


if __name__ == '__main__':
  absltest.main()
