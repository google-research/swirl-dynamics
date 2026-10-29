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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from swirl_dynamics.templates import models


class _MinimalModel(models.BaseModel):

  def initialize(self, rng: jax.Array) -> dict[str, dict[str, jax.Array]]:
    del rng
    return {"params": {"w": jnp.ones((2,))}}

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: jax.Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    del batch, rng
    loss = jnp.sum(jnp.square(params["w"]))
    return loss, ({"loss": loss}, mutables)


class ModelsTest(absltest.TestCase):

  def test_base_model_abstract_contract(self):
    with self.assertRaises(TypeError):
      models.BaseModel()  # pylint: disable=abstract-class-instantiated  # pyrefly: ignore[bad-instantiation]

    model = _MinimalModel()
    rng = jax.random.PRNGKey(0)
    variables = model.initialize(rng)
    loss, (metrics, _) = model.loss_fn(
        variables["params"], {"meta": "tag"}, rng, {}
    )
    self.assertAlmostEqual(float(loss), 2.0, places=6)
    self.assertIn("loss", metrics)
    with self.assertRaises(NotImplementedError):
      model.inference_fn(variables, "extra_arg", key="value")

    with self.assertRaises(NotImplementedError):
      model.eval_fn(variables, {}, rng)
    with self.assertRaises(NotImplementedError):
      model.inference_fn(variables)


if __name__ == "__main__":
  absltest.main()
