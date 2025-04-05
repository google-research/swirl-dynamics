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

"""Sampling with parallel denosing."""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import json
from typing import Literal

from absl import logging
import dask.array as da
from etils import epath
import flax.linen as nn
import jax
from jax.experimental import multihost_utils
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as pdfn
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import inference as inference_pipeline
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr
import xarray_tensorstore as xrts

P = jax.sharding.PartitionSpec

filesys = epath.backend.tf_backend


@dataclasses.dataclass
class TrajectorySamplerParallel:
  """Multi-diffusion sampler for long downscaled trajectory.

  This implements Algorithm 1 in https://arxiv.org/abs/2412.08079.

  This sampler is designed to sample a long trajectory, using samplers for short
  segments running in parallel. Each device is responsible for a single segment,
  with slight overlaps between contiguous segments. For example, if there are 4
  models each accounting for 3 days, with 1 day overlap, the total trajectory
  length will be 9 days:

                     day1  day2  day3  day4  day5  day6  day7  day8  day9
  model0 (device0)    x     x     x
  model1 (device1)                x     x     x
  model2 (device2)                            x     x     x
  model3 (device3)                                        x     x     x

  This results in non-unique values for the overlapped days (3, 5, 7 above). At
  each denoising step, these non-unique values are consolidated (by averaging
  their denoising targets) after each sampling step, through device-to-device
  communication (via `jax.lax.ppermute`), to ensure they become increasingly
  more consistent as the noise level decreases.

  At the end of the sampling process, the long sample is then denormalized by
  the corresponding climatology and added to the interpolated low-resolution
  dataset, so that the final sample is in original interpretable physical units.
  The final samples are saved to a Zarr store.

  Attributes:
    start_date: The start date of the trajectory to be sampled.
    total_sample_days: The total number of days to sample. This currently has a
      deterministic relationship with the number of devices used: `num_devices *
      (num_model_days - num_overlap_days) + num_overlap_days`.
    variables: Output variable names.
    lon_coords: The output longitude coordinates.
    lat_coords: The output latitude coordinates.
    scheme: The diffusion scheme (i.e. scale and sigma schedules).
    denoise_fn: The denoising function for a single segment, constructed from
      the trained denoising model with checkpoint loaded.
    tspan: The time steps for solving the sampling ODE or SDE.
    cond_loader: Dataloader that loads the conditions (i.e. the low-resolution
      inputs) in the sampling range.
    lowres_ds: The low-resolution input dataset, which is interpolated to
      high-resolution and added to the sampler output representing the residual
      values.
    stats_ds: The high-resolution statistics dataset, which contains the mean
      and standard deviation used to denormalize the sampled residuals before
      adding the interpolated low-resolution values.
    store_path: The Zarr path to store the final samples. Can be the direct save
      path (ends with '.zarr') or a directory ('/samples.zarr' will be
      appended).
    hour_interval: The hour interval to sample (e.g. every 2 hours). This is a
      property of the trained model.
    batch_size: The number of samples to generate in one batch.
    num_model_days: The number of days in a single sampling segment. This is a
      property of the trained model.
    num_overlap_days: The number of overlapping days between continguous sampled
      segments.
    sampling_type: The sampling type. Can be either "ode" or "sde".
    sde_noise_inflate: The SDE noise inflation.
    interp_method: The interpolation method for the low-resolution dataset. This
      must be consistent with the processing done to generate the training data.
    time_coords: The output time coordinates, derived from `start_date`,
      `total_sample_days`, and `hour_interval`.
  """

  start_date: str
  total_sample_days: int
  variables: Sequence[str]
  lon_coords: np.ndarray
  lat_coords: np.ndarray
  scheme: dfn_lib.Diffusion
  denoise_fn: dfn_lib.DenoiseFn
  tspan: jax.Array
  cond_loader: inference_pipeline.CondLoader
  lowres_ds: xr.Dataset
  stats_ds: xr.Dataset
  store_path: epath.PathLike
  hour_interval: int = 2
  batch_size: int = 1
  num_model_days: int = 3
  num_overlap_days: int = 1
  sampling_type: Literal["ode", "sde"] = "ode"
  sde_noise_inflate: float = 1.0
  interp_method: str = "linear"

  def __post_init__(self):
    expected_sample_days = (
        jax.device_count() * (self.num_model_days - self.num_overlap_days)
        + self.num_overlap_days
    )
    if self.total_sample_days != expected_sample_days:
      raise ValueError(
          f"`total_sample_days` ({self.total_sample_days}) expected to be equal"
          f" to {expected_sample_days} = num_devices ({jax.device_count()})"
          f" * (num_model_days ({self.num_model_days}) - num_overlap_days"
          f" ({self.num_overlap_days})) + num_overlap_days"
      )
    if not str(self.store_path).endswith(".zarr"):
      self.store_path = epath.Path(self.store_path) / "samples.zarr"

  @property
  def time_coords(self) -> np.ndarray:
    time_coords = np.datetime64(self.start_date) + np.arange(
        self.total_sample_days * (24 // self.hour_interval)
    ) * np.timedelta64(self.hour_interval, "h")
    # This is to avoid issues with pandas in xarray.
    return time_coords.astype("datetime64[ns]")

  def load_cond(self) -> dict[str, np.ndarray]:
    """Loads inputs for the entire trajectory."""
    # out shape ~ (samples, model_days * num_segments, lon, lat, variables)
    model_time_size = self.num_model_days * 24 // self.hour_interval
    interval = model_time_size - int(
        self.num_overlap_days * 24 // self.hour_interval
    )
    conds = []
    for i in range(0, len(self.time_coords), interval):
      days = self.time_coords[i : i + model_time_size].astype("datetime64[D]")
      cond = jax.tree.map(
          lambda x: jnp.stack([x] * self.batch_size),
          self.cond_loader.get_days(days=days),
      )
      conds.append(cond)

    # Note that the last one is excluded (goes over the boundary).
    conds = jax.tree.map(lambda *x: jnp.concatenate(x, axis=1), *conds[:-1])
    return conds

  @functools.cached_property
  def _consolidate(self):
    """Consolidates the overlaps by taking the average value."""

    overlap = self.num_overlap_days * 24 // self.hour_interval

    def _consolidate_fn(
        x: jax.Array,
        _left: jax.Array,  # pylint: disable=invalid-name
        _right: jax.Array,  # pylint: disable=invalid-name
        noise_inflate: bool = False,
    ) -> jax.Array:
      nsecs = jax.lax.psum(1, axis_name="segment")
      rhs, rhs_flag = jax.lax.ppermute(
          (x[:, :overlap], _left),
          axis_name="segment",
          perm=[(j, (j - 1) % nsecs) for j in range(nsecs)],
      )
      lhs, lhs_flag = jax.lax.ppermute(
          (x[:, -overlap:], _right),
          axis_name="segment",
          perm=[(j, (j + 1) % nsecs) for j in range(nsecs)],
      )
      inflate = jnp.sqrt(2.0) if noise_inflate else 1.0
      rhs = (1 - rhs_flag) * ((rhs + x[:, -overlap:]) / 2) * inflate + (
          rhs_flag * x[:, -overlap:]
      )
      lhs = (1 - lhs_flag) * ((lhs + x[:, :overlap]) / 2) * inflate + (
          lhs_flag * x[:, :overlap]
      )
      x = x.at[:, -overlap:].set(rhs)
      x = x.at[:, :overlap].set(lhs)
      return x

    return _consolidate_fn

  @functools.cached_property
  def initialize_state(self):
    """Initializes the diffusion state for the entire trajectory."""
    # out shape ~ (samples, time * num_segments, lon, lat, variables)

    @functools.partial(
        shard_map.shard_map,
        mesh=jax.make_mesh((jax.device_count(),), ("segment",)),
        in_specs=(P("segment"), P("segment"), P("segment")),
        out_specs=P(None, "segment", None, None, None),
    )
    def _shard_initialize(
        rng: jax.Array,
        _left: jax.Array,  # pylint: disable=invalid-name
        _right: jax.Array,  # pylint: disable=invalid-name
    ) -> jax.Array:
      shape = (
          self.batch_size,
          self.num_model_days * 24 // self.hour_interval,
          len(self.lon_coords),
          len(self.lat_coords),
          len(self.variables),
      )
      x1 = jax.random.normal(rng[0], shape)
      x1 = self._consolidate(x1, _left, _right, noise_inflate=True)
      return x1

    return jax.jit(_shard_initialize)

  @functools.cached_property
  def one_step_denoise(self):
    """Runs one step denoise."""

    in_specs = (
        P(None, "segment", None, None, None),  # x1
        P(),  # t1
        P(),  # t0
        P(None, "segment", None, None, None),  # cond
        P("segment"),  # rng
        P("segment"),  # _left
        P("segment"),  # _right
    )
    out_specs = P(None, "segment", None, None, None)

    @functools.partial(
        shard_map.shard_map,
        mesh=jax.make_mesh((jax.device_count(),), ("segment",)),
        in_specs=in_specs,
        out_specs=out_specs,
    )
    def _shard_denoise(
        x1: jax.Array,
        t1: jax.Array,
        t0: jax.Array,
        cond: Mapping[str, jax.Array],
        rng: jax.Array,
        _left: jax.Array,  # pylint: disable=invalid-name
        _right: jax.Array,  # pylint: disable=invalid-name
    ) -> jax.Array:
      # `_left` and `_right` indicates whether the shard is at the left or right
      # boundary of the domain.
      s1, sigma1 = self.scheme.scale(t1), self.scheme.sigma(t1)
      s0, sigma0 = self.scheme.scale(t0), self.scheme.sigma(t0)

      # Note: input is lon-lat but denoiser takes lat-lon.
      denoised = self.denoise_fn(
          jnp.rot90(x1, k=1, axes=(-3, -2)) / s1, sigma1, cond
      )
      denoised = jnp.rot90(denoised, k=-1, axes=(-3, -2))
      denoised = self._consolidate(denoised, _left, _right, noise_inflate=False)

      # ODE with exponential solver.
      if self.sampling_type == "ode":
        x0 = (
            s0 * sigma0 / (s1 * sigma1) * x1
            + s0 * (1 - sigma0 / sigma1) * denoised
        )
      # SDE with exponential solver.
      elif self.sampling_type == "sde":
        x0 = (
            s0 / s1 * (jnp.square(sigma0 / sigma1)) * x1
            + s0 * (1 - jnp.square(sigma0 / sigma1)) * denoised
        )
        noise = (
            s0
            * (sigma0 / sigma1)
            * jnp.sqrt(jnp.square(sigma1) - jnp.square(sigma0))
            # Shard map does not reduce dimension, so we need to get index 0.
            * jax.random.normal(rng[0], x0.shape)
        )
        # Averaging noise reduces its variance - we need to compensate it.
        noise = self._consolidate(noise, _left, _right, noise_inflate=True)
        noise = noise * self.sde_noise_inflate  # Additional noise inflation.
        x0 += noise
      else:
        raise ValueError(f"Unknown sampling type: {self.sampling_type}")
      return x0

    return jax.jit(_shard_denoise)

  def post_process_batch(self, state: jax.Array) -> dict[str, np.ndarray]:
    """Post-processes the final diffusion state."""
    state = multihost_utils.process_allgather(state)

    # Remove overlaps and concatenate.
    state = jnp.reshape(
        state,
        (
            self.batch_size,
            -1,
            self.num_model_days * 24 // self.hour_interval,
            len(self.lon_coords),
            len(self.lat_coords),
            len(self.variables),
        ),
    )
    d = (self.num_model_days - self.num_overlap_days) * 24 // self.hour_interval

    # Sanity check: overlaps are equal.
    overlap = self.num_overlap_days * 24 // self.hour_interval
    assert jnp.all(state[:, :-1, -overlap:] == state[:, 1:, :overlap])

    state0 = state[:, :-1, :d]
    state0 = jnp.reshape(
        state0,
        (
            self.batch_size,
            -1,
            len(self.lon_coords),
            len(self.lat_coords),
            len(self.variables),
        ),
    )
    samples = jnp.concatenate([state0, state[:, -1]], axis=1)

    # Interp low-resolution dataset.
    lowres_ds = (
        self.lowres_ds.interp(  # pytype: disable=wrong-arg-types
            coords={
                "longitude": self.lon_coords,
                "latitude": self.lat_coords,
            },
            method=self.interp_method,
        )
        .sel(time=self.time_coords.astype("datetime64[D]"))
        .transpose("time", "longitude", "latitude")
    )

    # Rescale and add interpolated low-res.
    time = xr.DataArray(self.time_coords, coords={"time": self.time_coords})
    ds = {}
    for i, v in enumerate(self.variables):
      var = np.asarray(samples[..., i], dtype=np.float32)
      var_stats = (
          self.stats_ds[v]
          .sel(dayofyear=time.dt.dayofyear, hour=time.dt.hour, drop=True)
          .transpose("stats", "time", "longitude", "latitude")
      )
      var_mean = var_stats.sel(stats="mean").to_numpy()
      var_std = var_stats.sel(stats="std").to_numpy()
      var_data = var * var_std + var_mean
      ds[v] = var_data + lowres_ds[v].to_numpy()
    return ds

  def initialize_stores(self, num_samples: int) -> None:
    """Initializes the sample stores."""
    ds = {}
    for v in self.variables:
      ds[v] = xr.DataArray(
          da.zeros((
              num_samples,
              len(self.time_coords),
              len(self.lon_coords),
              len(self.lat_coords),
          )),
          dims=["member", "time", "longitude", "latitude"],
          coords={
              "member": np.arange(num_samples),
              "time": self.time_coords,
              "longitude": self.lon_coords,
              "latitude": self.lat_coords,
          },
      )
    sample_ds = xr.Dataset(ds)

    # Initialize member chunk storage in the first process.
    if jax.process_index() == 0:
      logging.info("Initializing sample store %s", self.store_path)
      if filesys.exists(self.store_path):
        logging.info("Existing sample store found. Deleting it.")
        filesys.rmtree(self.store_path)

      sample_ds_pixel = sample_ds.chunk({  # member chunk
          "member": self.batch_size,
          "time": -1,
          "longitude": -1,
          "latitude": -1,
      })
      sample_ds_pixel.to_zarr(self.store_path, consolidated=True, compute=False)

  def save_samples(
      self, samples: dict[str, np.ndarray], batch_idx: int
  ) -> None:
    """Saves samples to zarr."""
    ds = {}
    for v in self.variables:
      ds[v] = xr.DataArray(
          samples[v],
          dims=["member", "time", "longitude", "latitude"],
          coords={
              "member": (
                  np.arange(self.batch_size) + batch_idx * self.batch_size
              ),
              "time": self.time_coords,
              "longitude": self.lon_coords,
              "latitude": self.lat_coords,
          },
      )
    sample_ds = xr.Dataset(ds)

    indexer = dict(
        member=slice(
            batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        ),
        time=slice(None),
        longitude=slice(None),
        latitude=slice(None),
    )

    if jax.process_index() == 0:  # Save in member chunk.
      sample_ds_member = sample_ds.chunk({  # Member chunk
          "member": self.batch_size,
          "time": -1,
          "longitude": -1,
          "latitude": -1,
      })
      sample_ds_member.to_zarr(self.store_path, compute=True, region=indexer)

  def generate_and_save(self, seed: int, num_samples: int) -> None:
    """Generate samples."""
    if num_samples % self.batch_size != 0:
      raise ValueError(
          f"Number of samples ({num_samples}) must be a multiple of batch size"
          f" ({self.batch_size})."
      )

    num_batches = num_samples // self.batch_size
    rng = jax.random.key(seed)
    self.initialize_stores(num_samples)

    _left = jnp.asarray([1.0] + [0.0] * (jax.device_count() - 1))  # pylint: disable=invalid-name
    _right = jnp.asarray([0.0] * (jax.device_count() - 1) + [1.0])  # pylint: disable=invalid-name

    # ~ (num_segments, batch_size, model_days, lon, lat, variables)
    cond = self.load_cond()

    for batch_idx in range(num_batches):

      logging.info("Sampling batch %d / %d", batch_idx + 1, num_batches)

      init_rng, *denoise_rng, rng = jax.random.split(
          rng, num=len(self.tspan) + 2
      )
      # ~ (num_segments, samples, time, lon, lat, variables)
      x = self.initialize_state(
          jax.random.split(init_rng, jax.device_count()), _left, _right
      )

      step = 0
      while step < len(self.tspan):
        logging.info("Denoising step %d / %d", step + 1, len(self.tspan))
        x = self.one_step_denoise(
            x,
            self.tspan[step],
            self.tspan[step + 1],
            cond,
            jax.random.split(denoise_rng[step], jax.device_count()),
            _left,
            _right,
        )
        step += 1

      logging.info("Post processing batch %d / %d", batch_idx + 1, num_batches)
      # dict[str, np.ndarray] ~ (time, batch_size, lon, lat)
      samples = self.post_process_batch(x)

      logging.info("Saving batch %d / %d to zarr", batch_idx + 1, num_batches)
      self.save_samples(samples, batch_idx)


