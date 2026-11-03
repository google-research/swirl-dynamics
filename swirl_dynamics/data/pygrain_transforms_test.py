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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import grain.python as pygrain
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.data import pygrain_sources
from swirl_dynamics.data import pygrain_transforms as transforms


class LinearRescaleTest(parameterized.TestCase):

  @parameterized.parameters(
      # test positive, negative and mixed output ranges
      {"output_range": (3, 4)},
      {"output_range": (-5, 5)},
      {"output_range": (-6, -3)},
  )
  def test_rescales_to_correct_range(self, output_range):
    input_max = 5
    raw_feature = np.arange(input_max)
    raw_sample = {"x": raw_feature}
    transform = transforms.LinearRescale(
        feature_name="x",
        input_range=(0, input_max - 1),
        output_range=output_range,
    )
    transformed_sample = transform.map(raw_sample)
    self.assertEqual(transformed_sample["x"].shape, (input_max,))
    self.assertEqual(np.max(transformed_sample["x"]), output_range[1])
    self.assertEqual(np.min(transformed_sample["x"]), output_range[0])

  @parameterized.parameters(
      {"input_range": (1, 0), "output_range": (3, 4)},
      {"input_range": (0, 1), "output_range": (-3, -3)},
  )
  def test_raises_invalid_range(self, input_range, output_range):
    with self.assertRaisesRegex(ValueError, "strictly smaller"):
      transforms.LinearRescale(
          feature_name="x", input_range=input_range, output_range=output_range
      )

  def test_preserves_float32_and_input(self):
    raw_feature = np.linspace(0, 64, 9, dtype=np.float32)
    raw_feature.flags.writeable = False  # e.g. a view from `ArrayDataSource`.
    transform = transforms.LinearRescale(
        feature_name="x", input_range=(0, 64), output_range=(-1, 1)
    )
    transformed_sample = transform.map({"x": raw_feature})
    self.assertEqual(transformed_sample["x"].dtype, np.float32)
    np.testing.assert_allclose(
        transformed_sample["x"], np.linspace(-1, 1, 9), atol=1e-6
    )
    np.testing.assert_array_equal(raw_feature, np.linspace(0, 64, 9))

  def test_float64_bounds_preserve_float32_feature_dtype(self):
    raw_feature = np.linspace(0.0, 64.0, 9, dtype=np.float32)
    transform = transforms.LinearRescale(
        feature_name="x",
        input_range=(np.float64(0.0), np.float64(64.0)),
        output_range=(np.float64(-1.0), np.float64(1.0)),
    )
    transformed = transform.map({"x": raw_feature})["x"]
    self.assertEqual(transformed.dtype, np.float32)
    np.testing.assert_allclose(
        transformed, np.linspace(-1.0, 1.0, 9, dtype=np.float32), atol=1e-6
    )


class NormalizeTest(parameterized.TestCase):

  def test_normalizes_to_correct_statistics(self):
    raw_feature = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
    )
    raw_sample = {"x": raw_feature}
    mean = np.mean(raw_feature, axis=0)
    std = np.std(raw_feature, axis=0)
    transform = transforms.Normalize(
        feature_name="x",
        mean=mean,
        std=std,
    )
    transformed_sample = transform.map(raw_sample)
    self.assertEqual(transformed_sample["x"].shape, raw_feature.shape)
    self.assertAlmostEqual(np.mean(transformed_sample["x"]), 0.0, delta=1e-5)
    self.assertAlmostEqual(np.std(transformed_sample["x"]), 1.0, delta=1e-5)

  def test_float64_stats_are_cast_to_float32_feature_dtype(self):
    rng = np.random.default_rng(0)
    raw_feature = rng.normal(size=(8, 5)).astype(np.float32)
    mean = rng.normal(size=(5,))  # float64
    std = rng.uniform(0.5, 2.0, size=(5,))  # float64
    transform = transforms.Normalize(feature_name="x", mean=mean, std=std)
    transformed = transform.map({"x": raw_feature})["x"]
    self.assertEqual(transformed.dtype, np.float32)
    # Statistics are rounded to float32 *before* the arithmetic (as TF did when
    # converting the constants to the tensor dtype), so the result is exactly
    # the float32 computation, not a rounded float64 one.
    expected = (raw_feature - mean.astype(np.float32)) / std.astype(np.float32)
    np.testing.assert_array_equal(transformed, expected)

  def test_jax_stats_give_numpy_result_equal_to_numpy_stats(self):
    rng = np.random.default_rng(1)
    raw_feature = rng.normal(size=(6, 4, 3)).astype(np.float32)
    np_mean = np.mean(raw_feature, axis=(0, 1))
    np_std = np.std(raw_feature, axis=(0, 1))
    jax_transform = transforms.Normalize(
        feature_name="x", mean=jnp.asarray(np_mean), std=jnp.asarray(np_std)
    )
    self.assertIsInstance(jax_transform.mean, np.ndarray)
    self.assertIsInstance(jax_transform.std, np.ndarray)
    np_transform = transforms.Normalize(
        feature_name="x", mean=np_mean, std=np_std
    )
    jax_result = jax_transform.map({"x": raw_feature.copy()})["x"]
    np_result = np_transform.map({"x": raw_feature.copy()})["x"]
    self.assertIsInstance(jax_result, np.ndarray)
    self.assertEqual(jax_result.dtype, np.float32)
    np.testing.assert_array_equal(jax_result, np_result)

  def test_integer_feature_uses_numpy_promotion(self):
    raw_feature = np.arange(6, dtype=np.int32).reshape(2, 3)
    transform = transforms.Normalize(feature_name="x", mean=1.5, std=2.0)
    transformed = transform.map({"x": raw_feature})["x"]
    self.assertEqual(transformed.dtype, np.float64)
    np.testing.assert_array_equal(transformed, (raw_feature - 1.5) / 2.0)


class RandomSectionTransformTest(parameterized.TestCase):

  @parameterized.parameters(
      {"num_steps": 4, "stride": 1},  # sample length < total length
      {"num_steps": 5, "stride": 2},  # sample length = total length
  )
  def test_correct_shapes_and_strides(self, num_steps, stride):
    raw_feature = np.tile(
        np.expand_dims(np.expand_dims(np.arange(0, 9), axis=1), axis=2),
        reps=(1, 16, 1),
    )  # shape = (9, 16, 1)
    raw_sample = {"u": raw_feature, "t": raw_feature}
    transform = transforms.RandomSection(
        feature_names=["u", "t"], num_steps=num_steps, stride=stride
    )
    self.assertEqual(transform.feature_names, ("u", "t"))
    transformed_sample = transform.random_map(
        raw_sample, rng=np.random.default_rng(23)
    )
    with self.subTest(checking="shapes"):
      self.assertEqual(transformed_sample["u"].shape, (num_steps, 16, 1))
      self.assertEqual(transformed_sample["t"].shape, (num_steps, 16, 1))
    with self.subTest(checking="stride"):
      self.assertEqual(
          transformed_sample["t"][1, 0, 0] - transformed_sample["t"][0, 0, 0],
          stride,
      )

  @parameterized.parameters(
      # sample length > total length with various strides
      {"num_steps": 10, "stride": 1},
      {"num_steps": 5, "stride": 2},
  )
  def test_raises_not_enough_steps(self, num_steps, stride):
    sample = {"u": np.zeros((8, 16, 1))}
    transform = transforms.RandomSection(
        feature_names=("u",), num_steps=num_steps, stride=stride
    )
    with self.assertRaisesRegex(ValueError, "Not enough steps"):
      transform.random_map(sample, rng=np.random.default_rng(23))

  def test_raises_unequal_feature_dim(self):
    sample = {"u": np.zeros((8, 16, 1)), "t": np.zeros((4, 16, 1))}
    transform = transforms.RandomSection(feature_names=("u", "t"), num_steps=5)
    with self.assertRaisesRegex(ValueError, "same dimension"):
      transform.random_map(sample, rng=np.random.default_rng(23))

  @parameterized.parameters(
      {"num_steps": 0, "stride": 1},
      {"num_steps": 2, "stride": 0},
  )
  def test_raises_non_positive_steps_or_stride(self, num_steps, stride):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      transforms.RandomSection(
          feature_names=("u",), num_steps=num_steps, stride=stride
      )

  @parameterized.parameters(
      {"feature_names": ()},
      {"feature_names": []},
      {"feature_names": "u"},
  )
  def test_raises_invalid_feature_names(self, feature_names):
    with self.assertRaisesRegex(ValueError, "non-empty sequence"):
      transforms.RandomSection(feature_names=feature_names, num_steps=2)

  def test_samples_shared_contiguous_strided_sections(self):
    total_length, num_steps, stride = 20, 4, 3
    sample_length = stride * (num_steps - 1) + 1
    u = np.arange(total_length, dtype=np.float32)[:, np.newaxis]
    t = 0.5 * np.arange(total_length, dtype=np.float32)
    x = np.ones((7, 1))
    transform = transforms.RandomSection(
        feature_names=("u", "t"), num_steps=num_steps, stride=stride
    )
    start_indices = set()
    for seed in range(200):
      out = transform.random_map(
          {"u": u, "t": t, "x": x}, rng=np.random.default_rng(seed)
      )
      start = int(out["u"][0, 0])
      start_indices.add(start)
      np.testing.assert_array_equal(
          out["u"], u[start : start + sample_length : stride]
      )
      # All sampled features share the same start index.
      np.testing.assert_array_equal(
          out["t"], t[start : start + sample_length : stride]
      )
      # Features that are not sampled are left untouched.
      self.assertIs(out["x"], x)
    # Every permissible start index is sampled.
    self.assertEqual(
        start_indices, set(range(total_length - sample_length + 1))
    )

  def test_same_rng_gives_same_section(self):
    u = np.arange(100)
    transform = transforms.RandomSection(feature_names=("u",), num_steps=5)
    out1 = transform.random_map({"u": u}, rng=np.random.default_rng(7))
    out2 = transform.random_map({"u": u}, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(out1["u"], out2["u"])


class SplitTransformTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          "split_sizes": (2, 2),
          "axis": 0,
          "split_names": ("u_0", "u_1"),
          "u0_shape": (2, 8),
      },
      {
          "split_sizes": (1, 3, 4),
          "axis": 1,
          "split_names": ("u_0", "u_1", "u_3"),
          "u0_shape": (4, 1),
      },
  )
  def test_splits_to_correct_shapes(
      self, split_sizes, split_names, axis, u0_shape
  ):
    sample = {"u": np.ones((4, 8))}
    transform = transforms.Split(
        feature_name="u",
        split_sizes=split_sizes,
        split_names=split_names,
        axis=axis,
    )
    transformed_sample = transform.map(sample)
    self.assertEqual(transformed_sample["u_0"].shape, u0_shape)

  def test_raises_length_mismatch(self):
    with self.assertRaisesRegex(ValueError, "Length .* must match"):
      transforms.Split(
          feature_name="u", split_sizes=(8,), split_names=("setup", "pred")
      )

  def test_keeps_presplit_feature(self):
    sample = {"u": np.zeros((4, 8))}
    transform = transforms.Split(
        feature_name="u",
        split_sizes=(4, 4),
        split_names=("u_setup", "u_pred"),
        axis=1,
        keep_presplit=True,
    )
    transformed_sample = transform.map(sample)
    self.assertIn("u", transformed_sample)

  def test_split_reusing_feature_name(self):
    sample = {"u": np.zeros((4, 8))}
    transform = transforms.Split(
        feature_name="u",
        split_sizes=(3, 5),
        split_names=("u", "u_rest"),
        axis=1,
        keep_presplit=False,
    )
    transformed_sample = transform.map(sample)
    self.assertIn("u", transformed_sample)
    self.assertEqual(transformed_sample["u"].shape, (4, 3))
    self.assertEqual(transformed_sample["u_rest"].shape, (4, 5))

  def test_splits_to_correct_values(self):
    u = np.arange(32).reshape(4, 8)
    transform = transforms.Split(
        feature_name="u",
        split_sizes=(1, 3, 4),
        split_names=("a", "b", "c"),
        axis=-1,
    )
    transformed_sample = transform.map({"u": u, "t": np.zeros(4)})
    self.assertSameElements(transformed_sample.keys(), ("a", "b", "c", "t"))
    np.testing.assert_array_equal(transformed_sample["a"], u[:, :1])
    np.testing.assert_array_equal(transformed_sample["b"], u[:, 1:4])
    np.testing.assert_array_equal(transformed_sample["c"], u[:, 4:])

  def test_raises_sizes_not_summing_to_dimension(self):
    transform = transforms.Split(
        feature_name="u", split_sizes=(2, 3), split_names=("a", "b"), axis=1
    )
    sample = {"u": np.zeros((4, 8))}
    with self.assertRaisesRegex(ValueError, "must sum to"):
      transform.map(sample)
    self.assertIn("u", sample)

  def test_raises_keep_presplit_name_collision(self):
    with self.assertRaisesRegex(ValueError, "keep_presplit=True"):
      transforms.Split(
          feature_name="u",
          split_sizes=(3, 5),
          split_names=("u", "u_rest"),
          keep_presplit=True,
      )


class SelectAsTransformTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          "select_features": "u",
          "as_features": "x",
          "u_shape": (2, 8),
      },
      {
          "select_features": "velocity",
          "as_features": "target",
          "u_shape": (4, 1),
      },
  )
  def test_renames_selected_features(
      self, select_features, as_features, u_shape
  ):
    sample = {
        select_features: np.ones(u_shape),
        "_dummy": np.ones(u_shape),
        "to_be_removed": np.ones(u_shape),
    }
    transform = transforms.SelectAs(
        select_features=select_features,
        as_features=as_features,
    )
    transformed_sample = transform.map(sample)
    self.assertIn(as_features, transformed_sample)
    self.assertIn("_dummy", transformed_sample)
    self.assertNotIn("to_be_removed", transformed_sample)

  def test_swaps_features(self):
    u, x = np.zeros(3), np.ones(5)
    transform = transforms.SelectAs(
        select_features=("u", "x"), as_features=("x", "u")
    )
    transformed_sample = transform.map({"u": u, "x": x, "_id": 7})
    self.assertIs(transformed_sample["x"], u)
    self.assertIs(transformed_sample["u"], x)
    self.assertEqual(transformed_sample["_id"], 7)

  @parameterized.parameters(
      {
          "select_features": ("u", "t"),
          "as_features": ("x",),
          "error_regex": "Length .* must match",
      },
      {
          "select_features": ("u", "u"),
          "as_features": ("x", "y"),
          "error_regex": "select_features.*must be unique",
      },
      {
          "select_features": ("u", "t"),
          "as_features": ("x", "x"),
          "error_regex": "as_features.*must be unique",
      },
  )
  def test_raises_invalid_features(
      self, select_features, as_features, error_regex
  ):
    with self.assertRaisesRegex(ValueError, error_regex):
      transforms.SelectAs(
          select_features=select_features, as_features=as_features
      )


class DataLoaderIntegrationTest(parameterized.TestCase):

  def _create_loader(self, source, seed, num_steps, stride, batch_size):
    return pygrain.DataLoader(
        data_source=source,
        sampler=pygrain.IndexSampler(
            num_records=len(source),
            shard_options=pygrain.NoSharding(),
            shuffle=True,
            num_epochs=None,
            seed=seed,
        ),
        operations=[
            transforms.RandomSection(
                feature_names=("u", "t"), num_steps=num_steps, stride=stride
            ),
            transforms.LinearRescale(
                feature_name="x", input_range=(0, 64), output_range=(-1, 1)
            ),
            pygrain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )

  def test_random_sections_in_dataloader(self):
    num_trajs, num_time, num_steps, stride, batch_size = 5, 30, 4, 2, 3
    sample_length = stride * (num_steps - 1) + 1
    # Values encode the trajectory and time indices to check exact windows.
    traj_idx, time_idx = np.meshgrid(
        np.arange(num_trajs), np.arange(num_time), indexing="ij"
    )
    u = (1000.0 * traj_idx + time_idx)[..., np.newaxis].astype(np.float32)
    t = 0.1 * time_idx.astype(np.float32)
    x = np.tile(np.linspace(0, 64, 8)[:, np.newaxis], (num_trajs, 1, 1))
    source = pygrain_sources.ArrayDataSource(u=u, t=t, x=x)

    loader = self._create_loader(source, 42, num_steps, stride, batch_size)
    batches = list(itertools.islice(loader, 8))

    for batch in batches:
      self.assertSameElements(batch.keys(), ("u", "t", "x"))
      self.assertEqual(batch["u"].shape, (batch_size, num_steps, 1))
      self.assertEqual(batch["t"].shape, (batch_size, num_steps))
      self.assertEqual(batch["x"].shape, (batch_size, 8, 1))
      self.assertTrue(np.all(np.abs(batch["x"]) <= 1))
      for b in range(batch_size):
        traj, start = divmod(int(batch["u"][b, 0, 0]), 1000)
        window = slice(start, start + sample_length, stride)
        np.testing.assert_array_equal(batch["u"][b], u[traj, window])
        np.testing.assert_array_equal(batch["t"][b], t[traj, window])

    with self.subTest("same_seed_same_batches"):
      loader2 = self._create_loader(source, 42, num_steps, stride, batch_size)
      for batch, batch2 in zip(
          batches, itertools.islice(loader2, 8), strict=True
      ):
        np.testing.assert_array_equal(batch["u"], batch2["u"])

    with self.subTest("different_seed_different_batches"):
      loader3 = self._create_loader(source, 43, num_steps, stride, batch_size)
      batches3 = list(itertools.islice(loader3, 8))
      self.assertFalse(
          all(
              np.array_equal(b["u"], b3["u"])
              for b, b3 in zip(batches, batches3, strict=True)
          )
      )


if __name__ == "__main__":
  absltest.main()
