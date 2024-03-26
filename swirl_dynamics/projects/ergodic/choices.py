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

"""Training choices.

enum.Enum classes that define the various experiment choices, such as system,
learned model integrator, neural architecture, etc.

References:

[1] Li, Zongyi, et al. "Fourier Neural Operator for Parametric Partial
  Differential Equations." International Conference on Learning
  Representations. 2020.
[2] Stachenfeld, Kimberly, et al. "Learned Coarse Models for Efficient
  Turbulence Simulation." arXiv e-prints (2021): arXiv-2112.
[3] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional
  networks for biomedical image segmentation." Medical image computing and
  computer-assisted intervention-MICCAI 2015: 18th international conference,
  Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer
  International Publishing, 2015.
[4] Kochkov, Dmitrii, et al. "Neural general circulation models." arXiv preprint
  arXiv:2311.07222 (2023).
"""

from collections.abc import Callable
import enum
import functools

from flax import linen as nn
import jax
import ml_collections
from swirl_dynamics.lib.networks import convnets
from swirl_dynamics.lib.networks import fno
from swirl_dynamics.lib.networks import nonlinear_fourier
from swirl_dynamics.lib.networks import unets
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.projects.ergodic import measure_distances
from swirl_dynamics.projects.ergodic import rollout_weighting


Array = jax.Array
MeasureDistFn = Callable[[Array, Array], float | Array]
RolloutWeightingFn = Callable[[int], Array]


class Experiment(enum.Enum):
  """Experiment choices.

  1. Lorenz 63 system (lorenz63).
  2. Kuramoto-Sivashinsky system, on a 1D grid (ks_1d).
  3. Navier-Stokes with Kolmogorov forcing, on a 2D grid (ns_2d).
  """

  L63 = "lorenz63"
  KS_1D = "ks_1d"
  NS_2D = "ns_2d"


class Integrator(enum.Enum):
  """Different ODE integrator choices."""

  EULER = "ExplicitEuler"
  RK4 = "RungeKutta4"
  ONE_STEP_DIRECT = "OneStepDirect"
  MULTI_STEP_DIRECT = "MultiStepDirect"

  def dispatch(
      self,
  ) -> ode.ScanOdeSolver | ode.MultiStepScanOdeSolver:
    """Dispatch integator.

    Returns:
      ScanOdeSolver | MultiStepScanOdeSolver
    """
    # TODO: Profile if the moveaxis call required here introduces a
    # bottleneck.
    return {
        "ExplicitEuler": ode.ExplicitEuler(time_axis_pos=1),
        "RungeKutta4": ode.RungeKutta4(time_axis_pos=1),
        "OneStepDirect": ode.OneStepDirect(time_axis_pos=1),
        "MultiStepDirect": ode.MultiStepDirect(time_axis_pos=1),
    }[self.value]


class MeasureDistance(enum.Enum):
  """Measure distance choices."""

  MMD = "MMD"  # Maximum mean discrepancy.
  MMD_DIST = "MMD_DIST"  # Distributed version of MMD.
  SD = "SD"  # Sinkhorn divergence.

  def dispatch(
      self,
      downsample_factor: int = 1,
  ) -> measure_distances.MeasureDistFn:
    """Dispatch measure distance.

    Args:
      downsample_factor: downsample factor for empirical distribution samples.

    Returns:
      Measure distance function.
    """
    dist_fn = {
        "MMD": measure_distances.mmd,
        "MMD_DIST": measure_distances.mmd_distributed,
        "SD": measure_distances.sinkhorn_div,
    }[self.value]
    if downsample_factor > 1:
      return functools.partial(
          measure_distances.spatial_downsampled_dist,
          dist_fn,
          spatial_downsample=downsample_factor,
      )
    return dist_fn


class Model(enum.Enum):
  """Model choices."""

  FNO = "Fno"  # One-dimensional Fourier Neural Operator [1].
  FNO_2D = "Fno2d"  # Two-dimensional Fourier Neural Operator [1].
  MLP = "MLP"  # Multi-layer perceptron.
  PERIODIC_CONV_NET_MODEL = "PeriodicConvNetModel"  # Dilated convolutions [2].
  UNET = "UNet"  # Regular UNet [3].

  def dispatch(self, conf: ml_collections.ConfigDict) -> nn.Module:
    """Dispatch model.

    Args:
      conf: Config dictionary.

    Returns:
      nn.Module
    """
    if self.value == Model.MLP.value:
      return nonlinear_fourier.MLP(features=conf.mlp_sizes, act_fn=nn.swish)
    if self.value == Model.PERIODIC_CONV_NET_MODEL.value:
      return convnets.PeriodicConvNetModel(
          latent_dim=conf.latent_dim,
          num_levels=conf.num_levels,
          num_processors=conf.num_processors,
          encoder_kernel_size=conf.encoder_kernel_size,
          decoder_kernel_size=conf.decoder_kernel_size,
          processor_kernel_size=conf.processor_kernel_size,
          padding=conf.padding,
          is_input_residual=conf.is_input_residual,
      )
    if self.value == Model.FNO.value:
      return fno.Fno(
          out_channels=conf.out_channels,
          hidden_channels=conf.hidden_channels,
          num_modes=conf.num_modes,
          lifting_channels=conf.lifting_channels,
          projection_channels=conf.projection_channels,
          num_blocks=conf.num_blocks,
          layers_per_block=conf.layers_per_block,
          block_skip_type=conf.block_skip_type,
          fft_norm=conf.fft_norm,
          separable=conf.separable,
      )
    if self.value == Model.FNO_2D.value:
      return fno.Fno2d(
          out_channels=conf.out_channels,
          num_modes=conf.num_modes,
          width=conf.width,
          fft_norm=conf.fft_norm,
      )
    if self.value == Model.UNET.value:
      return unets.UNet(
          out_channels=conf.out_channels,
          num_channels=conf.num_channels,
          downsample_ratio=conf.downsample_ratio,
          num_blocks=conf.num_blocks,
          padding=conf.padding,
          use_attention=conf.use_attention,
          use_position_encoding=conf.use_position_encoding,
          num_heads=conf.num_heads,
      )
    raise ValueError()


class RolloutWeighting(enum.Enum):
  """Rollout weighting choices.

  This is useful for stabilizing the rollout training, particularly when it is
  chaotic, see [4] for different choices
  """

  GEOMETRIC = "geometric"
  INV_SQRT = "inv_sqrt"
  INV_SQUARED = "inv_squared"
  LINEAR = "linear"
  NO_WEIGHT = "no_weight"

  def dispatch(self, conf: ml_collections.ConfigDict) -> RolloutWeightingFn:
    """Dispatch rollout weighting."""
    if self.value == RolloutWeighting.GEOMETRIC.value:
      return functools.partial(
          rollout_weighting.geometric,
          r=conf.rollout_weighting_r,
          clip=conf.rollout_weighting_clip
      )
    if self.value == RolloutWeighting.INV_SQRT.value:
      return functools.partial(
          rollout_weighting.inverse_sqrt,
          clip=conf.rollout_weighting_clip
      )
    if self.value == RolloutWeighting.INV_SQUARED.value:
      return functools.partial(
          rollout_weighting.inverse_squared,
          clip=conf.rollout_weighting_clip
      )
    if self.value == RolloutWeighting.LINEAR.value:
      return functools.partial(
          rollout_weighting.linear,
          m=conf.rollout_weighting_m,
          clip=conf.rollout_weighting_clip
      )
    if self.value == RolloutWeighting.NO_WEIGHT.value:
      return rollout_weighting.no_weight
    raise ValueError()
