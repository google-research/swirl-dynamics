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
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics.lib import diffusion as dfn_lib


class _DummyDenoiser(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, *, is_training):
    del sigma, cond, is_training
    return nn.Dense(self.out_channels)(x)


class _BatchNormDenoiser(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, *, is_training):
    del sigma, cond
    h = nn.BatchNorm(use_running_average=not is_training)(x)
    return nn.Dense(self.out_channels)(h)


class DenoisingTrainersTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(trainer_cls=dfn_lib.DenoisingTrainer, is_distributed=False),
      dict(
          trainer_cls=dfn_lib.DistributedDenoisingTrainer, is_distributed=True
      ),
  )
  def test_train_and_eval_steps(self, trainer_cls, is_distributed):
    input_shape = (8, 8, 1)
    cond_shape = {"channel:0": (4, 4, 1)}
    num_devices = jax.local_device_count() if is_distributed else 1
    batch_size = 2 * num_devices
    scheme = dfn_lib.Diffusion.create_variance_exploding(
        dfn_lib.tangent_noise_schedule()
    )
    model = dfn_lib.DenoisingModel(
        input_shape=input_shape,
        cond_shape=cond_shape,
        denoiser=_DummyDenoiser(out_channels=1),
        noise_sampling=dfn_lib.log_uniform_sampling(scheme),
        noise_weighting=dfn_lib.edm_weighting(),
    )
    trainer = trainer_cls(
        model=model,
        rng=jax.random.PRNGKey(1),
        optimizer=optax.adam(learning_rate=1e-4),
        ema_decay=0.9,
    )
    batch = {
        "x": np.ones((batch_size, *input_shape), dtype=np.float32),
        "cond": dfn_lib.cond_sample_from_shape(cond_shape, (batch_size,)),
    }
    data_iter = itertools.repeat(batch)

    train_metrics = trainer.train(data_iter, num_steps=2).compute()
    self.assertEqual(trainer.train_state.int_step, 2)
    self.assertIn("train_loss", train_metrics)
    self.assertIn("train_loss_std", train_metrics)
    self.assertTrue(np.isfinite(train_metrics["train_loss_std"]))
    expected_kernel_shape = (
        (num_devices, 1, 1) if is_distributed else (1, 1)
    )
    self.assertEqual(
        trainer.train_state.params["Dense_0"]["kernel"].shape,
        expected_kernel_shape,
    )

    eval_metrics = trainer.eval(data_iter, num_steps=2).compute()
    for i in range(model.num_eval_noise_levels):
      self.assertIn(f"eval_denoise_lvl{i}", eval_metrics)

  def test_ema_variables_and_inference_fn_with_mutables_and_mappings(self):
    input_shape = (8, 8, 1)
    batch_size = 2
    denoiser = _BatchNormDenoiser(out_channels=1)
    scheme = dfn_lib.Diffusion.create_variance_exploding(
        dfn_lib.tangent_noise_schedule()
    )
    model = dfn_lib.DenoisingModel(
        input_shape=input_shape,
        denoiser=denoiser,
        noise_sampling=dfn_lib.log_uniform_sampling(scheme),
        noise_weighting=dfn_lib.edm_weighting(),
    )
    trainer = dfn_lib.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(0),
        optimizer=optax.adam(learning_rate=1e-2),
        ema_decay=0.5,
    )
    batch = {"x": np.full((batch_size, *input_shape), 3.0, dtype=np.float32)}
    trainer.train(itertools.repeat(batch), num_steps=2)

    ema_vars = trainer.train_state.ema_variables
    self.assertIn("params", ema_vars)
    self.assertIn("batch_stats", ema_vars)

    # Mapping ema_state (e.g. restored from checkpoint as dict or FrozenDict).
    for mapping_state in (
        {"ema": ema_vars["params"]},
        flax.core.freeze({"ema": ema_vars["params"]}),
    ):
      restored_state = trainer.train_state.replace(ema_state=mapping_state)
      self.assertIn("batch_stats", restored_state.ema_variables)
      inf_fn = dfn_lib.DenoisingTrainer.inference_fn_from_state_dict(
          state=restored_state, denoiser=denoiser, use_ema=True
      )
      out = inf_fn(
          x=jnp.ones((batch_size, *input_shape)), sigma=jnp.ones((batch_size,))
      )
      self.assertEqual(out.shape, (batch_size, *input_shape))

    inf_non_ema = dfn_lib.DenoisingTrainer.inference_fn_from_state_dict(
        state=trainer.train_state, denoiser=denoiser, use_ema=False
    )
    out_non_ema = inf_non_ema(
        x=jnp.ones((batch_size, *input_shape)), sigma=jnp.ones((batch_size,))
    )
    self.assertEqual(out_non_ema.shape, (batch_size, *input_shape))

    no_ema = trainer.train_state.replace(ema_state=None)
    with self.assertRaisesRegex(ValueError, "EMA state is none"):
      _ = no_ema.ema_variables


if __name__ == "__main__":
  absltest.main()
