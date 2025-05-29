# Copyright 2025 The swirl_dynamics Authors.
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

"""Modules for evaluating generative downscaled climate samples."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
from skimage import filters
from swirl_dynamics.lib.metrics import probabilistic_forecast as prob_metrics
from swirl_dynamics.projects.probabilistic_diffusion import inference
from swirl_dynamics.templates import evaluate


# Duarte et al (2014), https://doi.org/10.1175/MWR-D-13-00368.1, page 4275.
_R_D = 287.058  # J / kg / K
_R_V = 461.85  # J / kg / K
_EPS = _R_D / _R_V  # Ratio of dry air to water vapor gas constants.
_C_VV = 1424.0  # J / kg / K
_C_VL = 4186.0  # J / kg / K
_L_V0 = 2.5e6  # J / kg
_T_REF = 273.15  # Zero degree Celsius in Kelvin, also Triple point temp.
_P_TRIPLE = 611.0  # Triple-point pressure of water, in Pa
_RAINWATER_DENSITY = 1000.0  # Rainwater density in [kg/m3]


def relative_humidity(*, t: jax.Array, q: jax.Array, p: jax.Array) -> jax.Array:
  """Computes the relative humidity with respect to liquid water.

  Args:
    t: Temperature in [K].
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].

  Returns:
    The relative humidity.
  """
  return vapor_pressure(q=q, p=p) / sat_vapor_pressure(t=t)


def sat_vapor_pressure(*, t: jax.Array) -> jax.Array:
  """Computes the saturation vapor pressure with respect to liquid water.

  The expression used here follows Duarte et al (2014),
  https://doi.org/10.1175/MWR-D-13-00368.1. All constants are defined in the
  same paper, page 4275.

  Args:
    t: Temperature in [K].

  Returns:
    The saturation vapor pressure in [Pa].
  """
  cpv = _C_VV + _R_V
  e0v = _L_V0 - _R_V * _T_REF
  alpha = (cpv - _C_VL) / _R_V
  beta = (e0v - (_C_VV - _C_VL) * _T_REF) / _R_V
  return (
      _P_TRIPLE * (t / _T_REF) ** alpha * jnp.exp(beta * (1 / _T_REF - 1 / t))
  )


def vapor_pressure(*, q: jax.Array, p: jax.Array) -> jax.Array:
  """Computes the vapor pressure from specific humidity and pressure.

  The expression used here can be obtained using the ideal gas law and Dalton's
  law:

                e = q*rho*Rv*T,      p=rho*Rm*T,

  where e is the vapor pressure, q is the specific humidity, rho is the air
  density, Rv is the gas constant for water vapor, Rm is the gas constant for
  moist air, and T is the air temperature. The gas constant for moist air can be
  written as:

                            Rm = Rv[eps + (1-eps)*q)],

  where eps=Rd/Rv, and the gas constants are Rv = 461.85 [J/kg/K],
  Rd = 287.058 [J/kg/K].

  Args:
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].

  Returns:
    The vapor pressure in [Pa].
  """
  return p * q / (_EPS + (1 - _EPS) * q)


def dewpoint_t(q: jax.Array, p: jax.Array) -> jax.Array:
  """Computes the dewpoint temperature from specific humidity and pressure.

  The expression is equation 7 in Lawrence (2005),
  https://doi.org/10.1175/BAMS-86-2-225.

  Args:
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].

  Returns:
    The dewpoint temperature in [K].
  """
  e = vapor_pressure(q=q, p=p)
  a1 = 17.625
  b1 = 243.04  # C
  c1 = 610.94  # Pa
  # We convert back to [K] from [C]
  return b1 * jnp.log(e / c1) / (a1 - jnp.log(e / c1)) + _T_REF


def sawti_lfp(
    *, t: jax.Array, q: jax.Array, p: jax.Array, u: jax.Array, v: jax.Array
) -> jax.Array:
  """Computes the Santa Ana Wind Threat Index (SAWTI) large fire potential.

  The expression is the weather component of the large fire potential, defined
  by Rolinski et al (2016; https://doi.org/10.1175/WAF-D-15-0141.1).

  Args:
    t: Air temperature in [K].
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].
    u: U-component of wind in [m/s].
    v: V-component of wind in [m/s].

  Returns:
    The SAWTI large fire potential in [K*m2/s2].
  """
  windspeed_sq = u * u + v * v
  dewpoint_depression = t - dewpoint_t(q=q, p=p)
  return 0.001 * windspeed_sq * dewpoint_depression


def fosberg_fire_weather_index(
    *, t: jax.Array, q: jax.Array, p: jax.Array, u: jax.Array, v: jax.Array
) -> jax.Array:
  """Computes the Fosberg Fire Weather Index (FWI).

  Taken from eq. 1 of Good rick (2002), https://doi.org/10.1071/WF02005. The
  index is restricted to be in the range [0, 100], following NOAA's SPC
  guidelines.

  Args:
    t: Air temperature in [K].
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].
    u: U-component of wind in [m/s].
    v: V-component of wind in [m/s].

  Returns:
    The Fosberg Fire Weather Index.
  """
  windspeed_sq = u * u + v * v
  # Convert from [m/s] to [mi/h]
  windspeed_sq = windspeed_sq * (3.6 * 0.621371) ** 2
  # Convert relative humidity from [0, 1] to [0, 100]
  h = relative_humidity(t=t, q=q, p=p) * 100
  # Convert temperature from [K] to [F]
  t_f = (t - _T_REF) * 1.8 + 32

  eq_moisture = jnp.where(
      h <= 10,
      0.03229 + 0.281073 * h - 0.000578 * h * t_f,  # <= 10%
      2.22749 + 0.160107 * h - 0.01478 * t_f,  # 10% < h <= 50%
  )
  eq_moisture = jnp.where(
      h > 50,
      21.0606 + 0.005565 * h**2 - 0.00035 * h * t_f - 0.483199 * h,  # > 50%
      eq_moisture,
  )

  eq_moisture = eq_moisture / 30

  moist_damping = (
      1 - 2 * eq_moisture + 1.5 * eq_moisture**2 - 0.5 * eq_moisture**3
  )
  ffwi = moist_damping * jnp.sqrt(1 + windspeed_sq) / 0.3002
  # Limiting the index to be in the range [0, 100]
  ffwi = jnp.where(ffwi > 100, 100.0, ffwi)
  return jnp.where(ffwi < 0, 0.0, ffwi)


def wbgt_abom(*, t: jax.Array, q: jax.Array, p: jax.Array) -> jax.Array:
  """Computes the simplified wet-bulb globe temperature (WBGT).

  Computes the WBGT from air temperature, specific humidity, and atmospheric
  pressure, following the simplified definition developed by the Australian
  Bureau of Meteorology. The definition can be found in equation 5 of Table 1
  in Willett & Sherwood (2012; https://doi.org/10.1002/joc.2257) in terms of
  the air temperature (in C) and vapor pressure (in hPa). We adapt it here to
  take SI units as input.

  Args:
    t: Air temperature in [K].
    q: Specific humidity in [kg/kg].
    p: Atmospheric pressure in [Pa].

  Returns:
    The WBGT in [Celsius].
  """
  # We divide the pressure term by 100 to accomodate pressure in Pa, not hPa.
  # We convert air temperature to [C] from [K].
  return 0.567 * (t - _T_REF) + (0.393 / 100) * vapor_pressure(q=q, p=p) + 3.94


def terrestrial_water_input(*, rain: jax.Array, snow: jax.Array) -> jax.Array:
  """Computes the terrestrial water input (TWI), from snow and rain rates.

  Computes the TWI from cumulative precipitation and the snow water equivalent
  accumulation, both estimated over the same time period. The definition
  can be found in Henn et al (2020; https://doi.org/10.1029/2020GL088189).
  The TWI is defined as the sum of rain and snow melt. Since both precipitation
  and snow water equivalent account for the snow rate, it is cancelled out by
  computing

  TWI = precipitation - snow_accumulation ~ (rain + snow) - (snow - snow_melt)
      =  rain + snow_melt.

  Here, we assume that other precipitation sources, such as hail and ice,
  are also included in the snow water equivalent accumulation, and therefore
  also cancel out.

  Snow accumulation is converted to depth by taking a constant rainwater density
  of 1000 kg/m2.

  Args:
    rain: Cumulative precipitation over a given time period. This is a positive
      number. Units are [mm].
    snow: Snow water equivalent accumulation over the same period. Negative
      values are physical and correspond to snow melting. Units are [kg/m2].

  Returns:
    The TWI in [mm].
  """
  # We convert snow water equivalent to [mm] from [kg/m2].
  snow_depth = snow / (_RAINWATER_DENSITY / 1000.0)
  return rain - snow_depth


def compute_derived_field(
    field_name: str, original_field_names: list[str], original_fields: jax.Array
) -> jax.Array:
  """Computes a derived field from an array of original fields.

  Args:
    field_name: The name of the requested derived field. This function only
      supports the computation of `RAIN`, `WINDSPEED10`, and `WBGT` at the
      moment.
    original_field_names: The name of the original fields, in order of their
      position in the last axis of `original_fields`.
    original_fields: The array of original fields, concatenated along the last
      axis.

  Returns:
    The derived field.

  Raises:
    ValueError: If the requested derived field is not supported.
  """
  if field_name == "RAIN":
    rainc_id = int(original_field_names.index("RAINC"))
    rainnc_id = int(original_field_names.index("RAINNC"))
    rain = original_fields[..., rainc_id] + original_fields[..., rainnc_id]
    return jnp.where(rain > 0.0, rain, 0.0)
  # We expect TWI fields to be named as "TWI_<period>", where <period> is the
  # time period over which the TWI is computed.
  elif "TWI_" in field_name:
    period = field_name.split("_")[-1]
    rain_id = int(original_field_names.index(f"RAIN_{period}"))
    snow_id = int(original_field_names.index(f"SNOW_{period}"))
    rain = original_fields[..., rain_id]
    snow = original_fields[..., snow_id]
    return terrestrial_water_input(rain=rain, snow=snow)
  elif field_name == "WINDSPEED10":
    u_id = int(original_field_names.index("U10"))
    v_id = int(original_field_names.index("V10"))
    return jnp.sqrt(
        original_fields[..., u_id] * original_fields[..., u_id]
        + original_fields[..., v_id] * original_fields[..., v_id]
    )
  elif field_name == "WBGT":
    t_id = int(original_field_names.index("T2"))
    q_id = int(original_field_names.index("Q2"))
    p_id = int(original_field_names.index("PSFC"))
    return wbgt_abom(
        t=original_fields[..., t_id],
        q=original_fields[..., q_id],
        p=original_fields[..., p_id],
    )
  elif field_name == "SAWTI":
    t_id = int(original_field_names.index("T2"))
    q_id = int(original_field_names.index("Q2"))
    p_id = int(original_field_names.index("PSFC"))
    u_id = int(original_field_names.index("U10"))
    v_id = int(original_field_names.index("V10"))
    return sawti_lfp(
        t=original_fields[..., t_id],
        q=original_fields[..., q_id],
        p=original_fields[..., p_id],
        u=original_fields[..., u_id],
        v=original_fields[..., v_id],
    )
  elif field_name == "RH":
    t_id = int(original_field_names.index("T2"))
    q_id = int(original_field_names.index("Q2"))
    p_id = int(original_field_names.index("PSFC"))
    return relative_humidity(
        t=original_fields[..., t_id],
        q=original_fields[..., q_id],
        p=original_fields[..., p_id],
    )
  elif field_name == "FFWI":
    t_id = int(original_field_names.index("T2"))
    q_id = int(original_field_names.index("Q2"))
    p_id = int(original_field_names.index("PSFC"))
    u_id = int(original_field_names.index("U10"))
    v_id = int(original_field_names.index("V10"))
    return fosberg_fire_weather_index(
        t=original_fields[..., t_id],
        q=original_fields[..., q_id],
        p=original_fields[..., p_id],
        u=original_fields[..., u_id],
        v=original_fields[..., v_id],
    )
  else:
    raise ValueError(f"Unsupported derived field: {field_name}.")


def get_derived_fields(
    original_fields: jax.Array,
    field_names: list[str],
) -> jax.Array | None:
  """Computes all non-existing trailing fields from a given list of fields.

  Args:
    original_fields: An array of fields, concatenated along the last axis.
    field_names: The name of all required fields, in order of their position in
      the last axis of `original_fields`. All fields in this list beyond the
      number of existing fields in `original_fields` are assumed to be derived
      fields.

  Returns:
    An array of derived fields, concatenated along the last axis of
    `original_fields`.
  """
  num_original_fields = original_fields.shape[-1]
  original_field_names = field_names[:num_original_fields]
  derived_field_names = field_names[num_original_fields:]
  derived_fields = []
  for field in derived_field_names:
    derived_fields.append(
        compute_derived_field(field, original_field_names, original_fields)
    )
  if derived_fields:
    return jnp.stack(derived_fields, axis=-1)


def _energy(
    x: jax.Array,
    window_type: str = "hann",
    axes: tuple[int, int] = (-2, -1),
    **kwargs,
) -> jax.Array:
  """Computes spectral energy of a tensor of 2D fields."""
  window_shape = tuple(x.shape[ax] for ax in sorted(axes))
  window = filters.window(window_type, window_shape)
  x_w = x * window
  return jnp.square(
      jnp.abs(
          jnp.fft.fftshift(
              jnp.fft.fft2(x_w, axes=axes, **kwargs), axes=axes, **kwargs
          )
      )
  )


def radial_spectra_fn(
    num_bins: int = 256,
    axes: tuple[int, int] = (-3, -2),
    resolution: float = 1,
    **kwargs,
) -> tuple[Callable[[jax.Array], jax.Array], np.ndarray]:
  """Returns a function to compute radial the spectral density of 2D fields.

  The returned function operates on square croppings of the input 2D fields.
  The cropping is obtained from the origin of each dimension to the required
  last index.

  The scaling factor in the energy spectrum is defined as
  `resolution / (4 * np.pi * num_bins**3)`, following Durran et al (2017):
  https://doi.org/10.1175/MWR-D-17-0056.1. This guarantees that the integral
  of the energy spectral density over the wavenumbers is equal to the integral
  over space of the input field squared.

  Args:
    num_bins: Number of bins used to digitize the 2D fields. The input image
      will be cropped in its spatial axes to be `num_bins` long.
    axes: Spatial dimension axes over which the spectrum is computed.
    resolution: The spatial resolution of the input field, specified in units
      per pixel, and considered isotropic here. The spatial resolution enters
      the scaling factor in the energy spectrum, and defines the units of the
      wavenumber. Consider the case where pixels are 9 km apart. Then, entering
      `resolution=9` will result in a spectrum with wavenumbers in units of
      `[1/km]`, and spectral densities in units of `[km] *
      [input_field_units]^2`. In this example, if the input was speed in
      `[m/s]`, the output spectrum will be in units of `[km*m^2/s^2]`.
    **kwargs: Keyword arguments passed to the _energy function.

  Returns:
    A function that computes the radial spectral density of input 2D fields, and
    the wavenumbers corresponding to the bins at which this spectral density is
    computed.
  """
  wavenum = np.fft.fftshift(np.fft.fftfreq(num_bins, d=resolution))
  # TODO: Enable computation over rectangular domains
  kx, ky = np.meshgrid(wavenum, wavenum)
  k = np.sqrt(kx**2 + ky**2)
  bins = np.linspace(np.min(k), np.max(k), num=num_bins)
  indices = np.digitize(k, bins)

  def _radial_spectra(x: jax.Array) -> jax.Array:
    """Computes radial spectra of 2D fields."""
    # Apply cropping to user-selected axes, and make them the trailing axes.
    x_cropped = x.swapaxes(axes[1], -1).swapaxes(axes[0], -2)[
        ..., :num_bins, :num_bins
    ]
    energy = _energy(x_cropped, axes=(-2, -1), **kwargs)
    energy_k = lambda kk: jnp.sum((indices == kk) * energy, axis=(-2, -1))
    # Make the trailing spatial axis the output mapping axis.
    rad_energy = jax.vmap(energy_k, in_axes=0, out_axes=axes[-1])(
        jnp.arange(1, num_bins)
    )
    # Apply scaling factor to the energy spectrum.
    rad_energy = rad_energy * resolution / (4 * np.pi * num_bins**3)
    return rad_energy

  return jax.jit(_radial_spectra), bins[1:]


def _unmasked(key_name: str) -> Callable[..., dict[str, Any]]:
  """Returns a function that removes the mask from a dict."""

  def unmasked_fn(**kwargs):
    return {key_name: kwargs[key_name], "mask": None}

  return unmasked_fn


def _masked(key_name: str) -> Callable[..., dict[str, Any]]:
  """Returns a function that uses the mask from a dict."""

  def masked_fn(**kwargs):
    return {key_name: kwargs[key_name], "mask": kwargs["mask"]}

  return masked_fn


def batch_inference(
    inference_fn: inference.CondSampler,
    samples_per_cond: int,
    batch: Mapping[str, Any],
    rng: jax.Array,
    sample_batch_size: int | None = None,
) -> jax.Array:
  """Runs batch inference with a conditional sampler given a single condition.

  Sampling is done in parallel across devices (single or multiple accelerators,
  with a single host).

  Args:
    inference_fn: A model inference function. See `inference.CondSampler` for
      the expected arguments.
    samples_per_cond: The number of samples to generate per condition.
    batch: A batch of data containing conditioning inputs (key "cond") and
      possibly guidance inputs (key "guidance_inputs").
    rng: A rng seed.
    sample_batch_size: The batch size used for conditional generation. If None,
      the number of devices is used.

  Returns:
    A batch of generated samples with shape [batch_size, samples_per_cond,
      *spatial_dims, channels].
  """
  num_devices = jax.local_device_count()
  samples_per_device = samples_per_cond // num_devices
  if sample_batch_size is None:
    sample_batch_size = num_devices
  device_batch_size = sample_batch_size // num_devices
  num_device_batches = samples_per_device // device_batch_size

  cond = batch["cond"]
  guidance_inputs = (
      batch["guidance_inputs"] if "guidance_inputs" in batch else None
  )

  def _batch_inference_per_device(rng: jax.Array) -> jax.Array:
    rngs = jax.random.split(rng, num=num_device_batches)

    def batch_inf_fn(rng: jax.Array) -> jax.Array:
      return inference_fn(
          num_samples=device_batch_size,
          cond=cond,
          guidance_inputs=guidance_inputs,
          rng=rng,
      )

    # using `jax.lax.map` instead of `jax.vmap` because the former is less
    # memory intensive and batch inference is expected to be very demanding.
    # ~ (num_device_batches, device_batch_size, *spatial_dims, channels)
    batched_samples = jax.lax.map(batch_inf_fn, rngs)
    # ~ (num_device_batches * device_batch_size, *spatial_dims, channels)
    return jnp.reshape(batched_samples, (-1,) + batched_samples.shape[2:])

  pmap_inference_fn = jax.pmap(
      _batch_inference_per_device, in_axes=0, out_axes=0
  )
  pmap_rngs = jax.random.split(rng, num=num_devices)
  # ~ (num_devices, samples_per_device, *spatial_dims, channels)
  samples = pmap_inference_fn(pmap_rngs)
  samples = jnp.reshape(samples, (1, -1) + samples.shape[2:])
  return samples


@dataclasses.dataclass(frozen=True)
class PairedDownscalingBenchmark(evaluate.Benchmark):
  """Draws conditional samples and evaluates probabilistic scores.

  Required `batch` schema::

    batch["cond"]: dict[str, jax.Array] | None  # a-priori condition
    batch["guidance_inputs"]: dict[str, jax.Array] | None  # guidance inputs
    batch["x"]: jax.Array  # observation wrt which the samples are evaluated

  NOTE: Batch size *should always be 1*.

  Attributes:
    num_samples_per_cond: The number of conditional samples to generate per
      condition. The samples are generated in batches.
    sample_batch_size: The batch size to generate conditional samples at.
    num_bins: The number of bins to use for the radial spectra.
    target_resolution: The target resolution of the downscaled samples, in km.
    field_names: The names of the fields to be evaluated. The fields are
      expected to be in the same order as the channels in the input and output
      data, but can exceed the number of channels in the batch. All additional
      fields are assumed to be derived fields, and computed on-the-fly before
      evaluating the metrics.
    landmask: The land mask, necessary for land-only metrics.
  """

  num_samples_per_cond: int
  sample_batch_size: int
  num_bins: int
  target_resolution: float
  field_names: list[str]
  landmask: np.ndarray

  def __post_init__(self):
    if self.num_samples_per_cond % self.sample_batch_size != 0:
      raise ValueError(
          f"`sample_batch_size` ({self.sample_batch_size}) must be divisible by"
          f" `num_samples_per_cond` ({self.num_samples_per_cond})."
      )
    num_devices = jax.local_device_count()
    if self.num_samples_per_cond % num_devices != 0:
      raise ValueError(
          f"`num_samples_per_cond` ({self.num_samples_per_cond}) must be"
          f" divisible by `num_devices` ({num_devices})."
      )
    if self.sample_batch_size % num_devices != 0:
      raise ValueError(
          f"`sample_batch_size` ({self.sample_batch_size}) must be divisible by"
          f" `num_devices` ({num_devices})."
      )

  def run_batch_inference(
      self,
      inference_fn: inference.CondSampler,
      batch: Mapping[str, Any],
      rng: jax.Array,
  ) -> jax.Array:
    """Runs batch inference with a conditional sampler.

    Sampling is done in parallel across devices.

    Args:
      inference_fn: A model inference function. See `inference.CondSampler` for
        the expected arguments.
      batch: A batch of data containing conditioning inputs (key "cond") and
        possibly guidance inputs (key "guidance_inputs").
      rng: A rng seed.

    Returns:
      A batch of generated samples with shape [batch_size, samples_per_cond,
        *spatial_dims, channels].
    """
    squeeze_fn = lambda x: jnp.squeeze(x, axis=0)
    squeezed_batch = {"cond": jax.tree.map(squeeze_fn, batch["cond"])}
    squeezed_batch["guidance_inputs"] = (
        batch["guidance_inputs"] if "guidance_inputs" in batch else None
    )
    return batch_inference(
        inference_fn,
        self.num_samples_per_cond,
        squeezed_batch,
        rng,
        sample_batch_size=self.sample_batch_size,
    )

  def compute_batch_metrics(
      self, pred: jax.Array, batch: Mapping[str, Any]
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Computes metrics on the batch.

    Results consist of collected types and aggregated types (see
    `swirl_dynamics.templates.Benchmark` protocol for their definitions and
    distinctions).

    The collected metrics consist of:
      * Channel-wise CRPS
      * Conditional standard deviations

    The aggregated metrics consist of:
      * Global mean CRPS (scalar)
      * Local mean CRPS (averaged for each location)

    Args:
      pred: The conditional samples generated by a model, with assumed shape (1,
        num_samples, *spatial, channels).
      batch: The evaluation batch data containing a reference observation, with
        assumed key `x`. batch[`x`] has shape (1, *spatial, channels).

    Returns:
      results_to_collect: Metrics to be collected in their current shape.
      results_to_aggregate: Metrics to aggregate.
    """
    obs = batch["x"]  # ~ (1, *spatial, channels)
    # Here we add the derived variables to obs and pred
    derived_obs = get_derived_fields(obs, self.field_names)
    if derived_obs is not None:
      obs = jnp.concatenate((obs, derived_obs), axis=-1)

    derived_pred = get_derived_fields(pred, self.field_names)
    if derived_pred is not None:
      pred = jnp.concatenate((pred, derived_pred), axis=-1)

    ens_var = jnp.var(pred, axis=1, ddof=1)  # ~ (1, *spatial, channels)
    ens_mean_mse = jnp.square(jnp.mean(pred, axis=1) - obs)
    # crps ~ (1, *spatial, channels)
    crps = prob_metrics.crps(pred, obs, direct_broadcast=False)
    bias = jnp.subtract(jnp.mean(pred, axis=1), obs)
    # Cast to exact shape to avoid issues with clu_metrics.
    mask = jnp.repeat(
        self.landmask[np.newaxis, :, :, np.newaxis], obs.shape[0], axis=0
    )
    mask = jnp.repeat(mask, obs.shape[-1], axis=-1)
    if mask.shape != obs.shape:
      raise ValueError(
          f"Mask shape {mask.shape} does not match target shape {obs.shape}."
      )

    rad_spec_fn, _ = radial_spectra_fn(
        self.num_bins, resolution=self.target_resolution
    )
    spectrum = rad_spec_fn(pred)  # ~ (1, ens_size, wavenumber, channels)
    ref_spectrum = rad_spec_fn(obs)  # ~ (1, wavenumber, channels)
    spectrum_log_ratio = jnp.mean(
        jnp.abs(jnp.log10(jnp.divide(spectrum, ref_spectrum))), axis=1
    )
    spectrum = jnp.mean(spectrum, axis=1)

    results_to_collect = dict(
        ens_var=ens_var,
        crps=crps,
        ens_mean_mse=ens_mean_mse,
    )
    results_to_aggregate = dict(
        ens_var=ens_var,
        crps=crps,
        ens_mean_mse=ens_mean_mse,
        bias=bias,
        radial_spectrum=spectrum,
        obs_radial_spectrum=ref_spectrum,
        radial_spectrum_log_ratio=spectrum_log_ratio,
        mask=mask,
    )
    return results_to_collect, results_to_aggregate


class PairedDownscalingEvaluator(evaluate.Evaluator):
  """Evaluator for the conditional sampling benchmark."""

  @flax.struct.dataclass
  class AggregatingMetrics(clu_metrics.Collection):
    """Aggregated metric definitions for the PairedDownscalingEvaluator."""

    # Time-averaged spatial metrics
    crps: (
        evaluate.TensorAverage(axis=0)
        .from_output("crps")
        .from_fun(_unmasked("crps"))
    )
    bias: (
        evaluate.TensorAverage(axis=0)
        .from_output("bias")
        .from_fun(_unmasked("bias"))
    )
    ens_mean_mse: (
        evaluate.TensorAverage(axis=0)
        .from_output("ens_mean_mse")
        .from_fun(_unmasked("ens_mean_mse"))
    )
    ens_variance: (
        evaluate.TensorAverage(axis=0)
        .from_output("ens_var")
        .from_fun(_unmasked("ens_var"))
    )
    # Spread skill ratio (https://doi.org/10.1175/JHM-D-14-0008.1)
    variance_mse_ratio: evaluate.TensorRatio(axis=0).from_outputs(
        "ens_var", "ens_mean_mse"
    )

    # Time-averaged spectra
    radial_spectrum: (
        evaluate.TensorAverage(axis=0)
        .from_output("radial_spectrum")
        .from_fun(_unmasked("radial_spectrum"))
    )
    obs_radial_spectrum: (
        evaluate.TensorAverage(axis=0)
        .from_output("obs_radial_spectrum")
        .from_fun(_unmasked("obs_radial_spectrum"))
    )
    radial_spectrum_log_ratio: (
        evaluate.TensorAverage(axis=0)
        .from_output("radial_spectrum_log_ratio")
        .from_fun(_unmasked("radial_spectrum_log_ratio"))
    )

    # Time and space averaged metrics
    global_mean_crps: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("crps")
        .from_fun(_unmasked("crps"))
    )
    global_mean_bias: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("bias")
        .from_fun(_unmasked("bias"))
    )
    global_mean_ens_mean_mse: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("ens_mean_mse")
        .from_fun(_unmasked("ens_mean_mse"))
    )
    global_mean_ens_variance: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("ens_var")
        .from_fun(_unmasked("ens_var"))
    )
    global_mean_variance_mse_ratio: evaluate.TensorRatio(
        axis=(0, 1, 2)
    ).from_outputs("ens_var", "ens_mean_mse")

    # Land-only metrics
    land_mean_crps: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("crps")
        .from_fun(_masked("crps"))
    )
    land_mean_bias: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("bias")
        .from_fun(_masked("bias"))
    )
    land_mean_ens_mean_mse: (
        evaluate.TensorAverage(axis=(0, 1, 2))
        .from_output("ens_mean_mse")
        .from_fun(_masked("ens_mean_mse"))
    )

  @property
  def scalar_metrics_to_log(self) -> dict[str, jax.Array]:
    """Logs global scalar metrics."""
    scalar_metrics = {}
    agg_metrics = self.state.compute_aggregated_metrics()
    for model_key, metric_dict in agg_metrics.items():
      scalar_metrics[f"{model_key}/global_mean_crps"] = jnp.mean(
          metric_dict["global_mean_crps"]
      )
      scalar_metrics[f"{model_key}/global_mean_ens_mean_mse"] = jnp.mean(
          metric_dict["global_mean_ens_mean_mse"]
      )
      scalar_metrics[f"{model_key}/global_mean_ens_variance"] = jnp.mean(
          metric_dict["global_mean_ens_variance"]
      )
    return scalar_metrics
