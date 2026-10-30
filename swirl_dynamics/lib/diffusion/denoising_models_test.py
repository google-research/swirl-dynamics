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

from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib


class _SimpleDenoiser(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del is_training
    h = nn.Dense(self.features)(x)
    out = jax.vmap(jnp.multiply, in_axes=(0, 0))(h, sigma)
    if cond:
      out = jax.vmap(jnp.add, in_axes=(0, 0))(out, cond["bias"])
    return out


class _BatchNormDenoiser(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma, cond
    h = nn.BatchNorm(use_running_average=not is_training)(x)
    return nn.Dense(self.features)(h)


class _MappingCondDenoiser(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma, is_training
    if cond is None:
      raise ValueError("Expected non-None cond mapping when cond_shape={}.")
    return nn.Dense(self.features)(x)


class DenoisingModelsTest(parameterized.TestCase):

  def test_cond_sample_from_shape_supports_mappings(self):
    self.assertIsNone(dfn_lib.cond_sample_from_shape(None))
    arr = dfn_lib.cond_sample_from_shape(cast(Any, (4, 2)), batch_dims=(3,))
    self.assertEqual(arr.shape, (3, 4, 2))

    frozen = flax.core.freeze({"a": (4, 1), "nested": {"b": (2, 2)}})
    sample = dfn_lib.cond_sample_from_shape(frozen, batch_dims=(2,))
    self.assertEqual(sample["a"].shape, (2, 4, 1))
    self.assertEqual(sample["nested"]["b"].shape, (2, 2, 2))

    with self.assertRaisesRegex(TypeError, "Cannot initialize shape"):
      dfn_lib.cond_sample_from_shape(cast(Any, [4, 2]))

  @parameterized.parameters(
      dict(input_shape=(16, 1), cond_shape=None),
      dict(input_shape=(8, 8, 1), cond_shape={"bias": (8, 8, 3)}),
  )
  def test_denoising_model_loss_eval_and_inference(
      self, input_shape, cond_shape
  ):
    batch_size = 3
    scheme = dfn_lib.Diffusion.create_variance_exploding(
        dfn_lib.tangent_noise_schedule()
    )
    denoiser = _SimpleDenoiser(features=3 if cond_shape else 1)
    model = dfn_lib.DenoisingModel(
        input_shape=input_shape,
        cond_shape=cond_shape,
        denoiser=denoiser,
        noise_sampling=dfn_lib.log_uniform_sampling(scheme),
        noise_weighting=dfn_lib.edm_weighting(),
        num_eval_noise_levels=4,
        num_eval_cases_per_lvl=2,
    )
    variables = model.initialize(jax.random.PRNGKey(0))
    batch = {
        "x": np.ones((batch_size,) + input_shape, dtype=np.float32),
        "cond": dfn_lib.cond_sample_from_shape(cond_shape, (batch_size,)),
    }
    loss, (train_metric, _) = jax.jit(model.loss_fn)(
        params=variables["params"],
        batch=batch,
        rng=jax.random.PRNGKey(1),
        mutables={},
    )
    self.assertTrue(np.isfinite(loss))
    self.assertIn("loss", train_metric)

    eval_metric = jax.jit(model.eval_fn)(
        variables, batch, jax.random.PRNGKey(2)
    )
    self.assertLen(eval_metric, 4)
    for i in range(4):
      self.assertIn(f"sigma_lvl{i}", eval_metric)

    denoise_fn = dfn_lib.DenoisingModel.inference_fn(
        variables=variables, denoiser=denoiser
    )
    out_scalar = jax.jit(denoise_fn)(
        jnp.ones((batch_size,) + input_shape),
        0.5,
        batch["cond"],
    )
    out_vec = jax.jit(denoise_fn)(
        jnp.ones((batch_size,) + input_shape),
        jnp.full((batch_size,), 0.5),
        batch["cond"],
    )
    expected_out_shape = (batch_size,) + input_shape[:-1] + (
        3 if cond_shape else 1,
    )
    self.assertEqual(out_scalar.shape, expected_out_shape)
    np.testing.assert_allclose(out_scalar, out_vec, rtol=1e-6)

  def test_denoising_model_loss_fn_threads_mutables(self):
    input_shape = (8, 1)
    scheme = dfn_lib.Diffusion.create_variance_exploding(
        dfn_lib.tangent_noise_schedule()
    )
    model = dfn_lib.DenoisingModel(
        input_shape=input_shape,
        denoiser=_BatchNormDenoiser(features=1),
        noise_sampling=dfn_lib.log_uniform_sampling(scheme),
        noise_weighting=dfn_lib.edm_weighting(),
    )
    variables = model.initialize(jax.random.PRNGKey(0))
    init_mean = variables["batch_stats"]["BatchNorm_0"]["mean"]
    batch = {"x": np.full((4,) + input_shape, 5.0, dtype=np.float32)}
    loss, (_, updated_mutables) = jax.jit(model.loss_fn)(
        params=variables["params"],
        batch=batch,
        rng=jax.random.PRNGKey(3),
        mutables={"batch_stats": variables["batch_stats"]},
    )
    self.assertTrue(np.isfinite(loss))
    self.assertFalse(
        np.allclose(
            init_mean, updated_mutables["batch_stats"]["BatchNorm_0"]["mean"]
        )
    )

  def test_denoising_model_empty_cond_shape_passes_mapping(self):
    input_shape = (8, 1)
    scheme = dfn_lib.Diffusion.create_variance_exploding(
        dfn_lib.tangent_noise_schedule()
    )
    model = dfn_lib.DenoisingModel(
        input_shape=input_shape,
        cond_shape={},
        denoiser=_MappingCondDenoiser(features=1),
        noise_sampling=dfn_lib.log_uniform_sampling(scheme),
        noise_weighting=dfn_lib.edm_weighting(),
        num_eval_noise_levels=2,
    )
    variables = model.initialize(jax.random.PRNGKey(0))
    batch = {"x": np.ones((3,) + input_shape, dtype=np.float32), "cond": {}}
    loss, _ = jax.jit(model.loss_fn)(
        params=variables["params"],
        batch=batch,
        rng=jax.random.PRNGKey(1),
        mutables={},
    )
    self.assertTrue(np.isfinite(loss))
    eval_metrics = jax.jit(model.eval_fn)(
        variables, batch, jax.random.PRNGKey(2)
    )
    self.assertLen(eval_metrics, 2)


if __name__ == "__main__":
  absltest.main()
