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

from absl.testing import absltest
from absl.testing import parameterized
from swirl_dynamics.data import tfgrain_transforms as transforms
import tensorflow as tf


class LinearRescaleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      # test positive, negative and mixed output ranges
      {"output_range": (3, 4)},
      {"output_range": (-5, 5)},
      {"output_range": (-6, -3)},
  )
  def test_rescales_to_correct_range(self, output_range):
    input_max = 5
    raw_feature = tf.range(input_max)
    raw_sample = {"x": raw_feature}
    transform = transforms.LinearRescale(
        feature_name="x",
        input_range=(0, input_max - 1),
        output_range=output_range,
    )
    transformed_sample = transform.map(raw_sample)
    self.assertEqual(transformed_sample["x"].shape, (input_max,))
    self.assertEqual(tf.reduce_max(transformed_sample["x"]), output_range[1])
    self.assertEqual(tf.reduce_min(transformed_sample["x"]), output_range[0])

  @parameterized.parameters(
      {"input_range": (1, 0), "output_range": (3, 4)},
      {"input_range": (0, 1), "output_range": (-3, -3)},
  )
  def test_raises_invalid_range(self, input_range, output_range):
    with self.assertRaisesRegex(ValueError, "strictly smaller"):
      transforms.LinearRescale(
          feature_name="x", input_range=input_range, output_range=output_range
      )


class NormalizeTest(tf.test.TestCase, parameterized.TestCase):

  def test_normalizes_to_correct_statistics(self):
    raw_feature = tf.constant(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    )
    raw_sample = {"x": raw_feature}
    mean = tf.math.reduce_mean(raw_feature, axis=0)
    std = tf.math.reduce_std(raw_feature, axis=0)
    transform = transforms.Normalize(
        feature_name="x",
        mean=mean,
        std=std,
    )
    transformed_sample = transform.map(raw_sample)
    self.assertEqual(transformed_sample["x"].shape, raw_feature.shape)
    self.assertNear(tf.math.reduce_mean(transformed_sample["x"]), 0.0, 1e-5)
    self.assertNear(tf.math.reduce_std(transformed_sample["x"]), 1.0, 1e-5)


class RandomSectionTransformTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {"num_steps": 4, "stride": 1},  # sample length < total length
      {"num_steps": 5, "stride": 2},  # sample length = total length
  )
  def test_correct_shapes_and_strides(self, num_steps, stride):
    raw_feature = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.range(0, 9), axis=1), axis=2),
        multiples=(1, 16, 1),
    )  # shape = (9, 16, 1)
    raw_sample = {"u": raw_feature, "t": raw_feature}
    transform = transforms.RandomSection(
        feature_names=("u", "t"), num_steps=num_steps, stride=stride
    )
    transformed_sample = transform.random_map(
        raw_sample, seed=tf.constant((2, 3))
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
    sample = {"u": tf.zeros((8, 16, 1))}
    transform = transforms.RandomSection(
        feature_names=("u",), num_steps=num_steps, stride=stride
    )
    with self.assertRaisesRegex(ValueError, "Not enough steps"):
      transform.random_map(sample, seed=tf.constant((2, 3)))

  def test_raises_unequal_feature_dim(self):
    sample = {"u": tf.zeros((8, 16, 1)), "t": tf.zeros((4, 16, 1))}
    transform = transforms.RandomSection(feature_names=("u", "t"), num_steps=5)
    with self.assertRaisesRegex(ValueError, "same dimension"):
      transform.random_map(sample, seed=tf.constant((2, 3)))


class SplitTransformTest(tf.test.TestCase, parameterized.TestCase):

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
    sample = {"u": tf.ones((4, 8))}
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
    sample = {"u": tf.zeros((4, 8))}
    transform = transforms.Split(
        feature_name="u",
        split_sizes=(4, 4),
        split_names=("u_setup", "u_pred"),
        axis=1,
        keep_presplit=True,
    )
    transformed_sample = transform.map(sample)
    self.assertIn("u", transformed_sample)


if __name__ == "__main__":
  absltest.main()
