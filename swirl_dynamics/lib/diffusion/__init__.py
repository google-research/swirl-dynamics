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

"""Diffusion library."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from swirl_dynamics.lib.diffusion.denoising_models import (
    CondDict,
    DenoisingFlaxModule,
    DenoisingModel,
    ShapeDict,
    cond_sample_from_shape,
)
from swirl_dynamics.lib.diffusion.diffusion import (
    Diffusion,
    InvertibleSchedule,
    NoiseLevelSampling,
    NoiseLossWeighting,
    create_variance_exploding_scheme,
    create_variance_preserving_scheme,
    edm_weighting,
    exponential_noise_schedule,
    inverse_squared_weighting,
    log_normal_sampling,
    log_uniform_sampling,
    min_snr_edm_weighting,
    power_noise_schedule,
    t_edm_weighting,
    tangent_noise_schedule,
    time_uniform_sampling,
)
from swirl_dynamics.lib.diffusion.guidance import (
    ClassifierFreeContrastive,
    ClassifierFreeHybrid,
    InfillFromSlices,
    InterlockingFrames,
    Transform as GuidanceTransform,
)
from swirl_dynamics.lib.diffusion.preconditioning import (
    Preconditioned,
    compute_denoising_preconditioners,
    precondition,
)
from swirl_dynamics.lib.diffusion.samplers import (
    DenoiseFn,
    ExponentialOdeSampler,
    ExponentialSdeSampler,
    NoiseDist,
    OdeSampler,
    Sampler,
    ScoreFn,
    SdeSampler,
    TStudentOdeSampler,
    TStudentSdeSampler,
    TimeStepScheduler,
    edm_noise_decay,
    exponential_noise_decay,
    uniform_time,
)
from swirl_dynamics.lib.diffusion.unets import (
    AxialMLPInterpConvMerge,
    EmbConvMerge,
    HeavyTailedDenoiser,
    InterpConvMerge,
    MergeChannelCond,
    PreconditionedDenoiser as PreconditionedDenoiserUNet,
    UNet,
)
from swirl_dynamics.lib.diffusion.unets3d import PreconditionedDenoiser3d as PreconditionedDenoiserUNet3d
from swirl_dynamics.lib.diffusion.unets3d import UNet3d
from swirl_dynamics.lib.diffusion.vivit import ViViT
from swirl_dynamics.lib.diffusion.vivit_diffusion import PreconditionedDenoiser as PreconditionedDenoiserViViT
from swirl_dynamics.lib.diffusion.vivit_diffusion import ViViTDiffusion

HeavyTailedDenoiserUNet = HeavyTailedDenoiser

__all__ = [
    "AxialMLPInterpConvMerge",
    "ClassifierFreeContrastive",
    "ClassifierFreeHybrid",
    "CondDict",
    "DenoiseFn",
    "DenoisingFlaxModule",
    "DenoisingModel",
    "Diffusion",
    "EmbConvMerge",
    "ExponentialOdeSampler",
    "ExponentialSdeSampler",
    "GuidanceTransform",
    "HeavyTailedDenoiser",
    "HeavyTailedDenoiserUNet",
    "InfillFromSlices",
    "InterlockingFrames",
    "InterpConvMerge",
    "InvertibleSchedule",
    "MergeChannelCond",
    "NoiseDist",
    "NoiseLevelSampling",
    "NoiseLossWeighting",
    "OdeSampler",
    "Preconditioned",
    "PreconditionedDenoiserUNet",
    "PreconditionedDenoiserUNet3d",
    "PreconditionedDenoiserViViT",
    "Sampler",
    "ScoreFn",
    "SdeSampler",
    "ShapeDict",
    "TStudentOdeSampler",
    "TStudentSdeSampler",
    "TimeStepScheduler",
    "UNet",
    "UNet3d",
    "ViViT",
    "ViViTDiffusion",
    "compute_denoising_preconditioners",
    "cond_sample_from_shape",
    "create_variance_exploding_scheme",
    "create_variance_preserving_scheme",
    "edm_noise_decay",
    "edm_weighting",
    "exponential_noise_decay",
    "exponential_noise_schedule",
    "inverse_squared_weighting",
    "log_normal_sampling",
    "log_uniform_sampling",
    "min_snr_edm_weighting",
    "power_noise_schedule",
    "precondition",
    "t_edm_weighting",
    "tangent_noise_schedule",
    "time_uniform_sampling",
    "uniform_time",
]
