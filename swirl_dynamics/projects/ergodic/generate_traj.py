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

"""Generate predict trajectory.

Helper script for generating trajectories from pre-trained models.
"""

import functools
import json
from os import path as osp

from absl import app
from absl import flags
from jax import numpy as jnp
import numpy as np
from orbax import checkpoint
import pandas as pd
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.projects.ergodic import choices
import tensorflow as tf
import tqdm.auto as tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_dir", None, "Path to experiment with trained models.")


def create_train_name(args_dict):
  """Create name from args."""
  model_display_dict = {
      "PeriodicConvNetModel": "CNN",
      "Fno": "FNO",
      "Fno2d": "MNO",
  }
  use_curriculum = args_dict["use_curriculum"]
  use_pushfwd = args_dict["use_pushfwd"]
  measure_dist_type = args_dict["measure_dist_type"]
  measure_dist_lambda = args_dict["measure_dist_lambda"]
  measure_dist_k_lambda = args_dict["measure_dist_k_lambda"]
  model_name = args_dict["model"]
  if use_curriculum:
    if use_pushfwd:
      train_name = "Pfwd"
    else:
      train_name = "Curr"
  else:
    train_name = "1-step"
  train_name_base = train_name
  train_name += f" {model_display_dict[model_name]}"
  if measure_dist_lambda > 0.0 or measure_dist_k_lambda > 0.0:
    train_name += f" {measure_dist_type}"
    train_name += f" λ1={int(measure_dist_lambda)}"
    train_name += f", λ2={int(measure_dist_k_lambda)}"
  return train_name_base, train_name


def parse_dir(exp_dir):
  """Parse directory to load arguments json."""
  exps = {}
  cnt = 0
  if tf.io.gfile.exists(exp_dir):
    dirs = tf.io.gfile.listdir(exp_dir)
  else:
    raise FileNotFoundError(f"Could not list directory: {exp_dir}.")
  for d in dirs:
    if tf.io.gfile.exists(osp.join(exp_dir, d, "config.json")):
      with tf.io.gfile.GFile(osp.join(exp_dir, d, "config.json"), "r") as f:
        args = json.load(f)
        if isinstance(args, str):
          args = json.loads(args)
      train_name_base, train_name = create_train_name(args)
      args["ckpt_path"] = osp.join(exp_dir, d, "checkpoints")
      args["traj_path"] = osp.join(exp_dir, d, "trajectories")
      args["train_name_base"] = train_name_base
      args["train_name"] = train_name
      cnt += 1
      exps[cnt] = args
    else:
      continue
  return pd.DataFrame.from_dict(exps, orient="index").sort_index()


def generate_pred_traj(exps_df, all_steps, dt, trajs, mean=None, std=None):
  """Generate predicted trajectories and save to file."""
  pbar = tqdm.tqdm(exps_df.iterrows(), total=len(exps_df), desc="Exps")

  for r in pbar:
    first_step = r[1]["save_interval_steps"]
    total_steps = r[1]["train_steps"]
    save_every = r[1]["save_interval_steps"]

    train_name = r[1]["train_name"]
    seed = r[1]["seed"]
    ckpt_dir = r[1]["ckpt_path"]
    integrator_choice = r[1]["integrator"]
    model_choice = r[1]["model"]
    normalize = r[1]["normalize"]
    batch_size = r[1]["batch_size"]
    ckpt_pbar = tqdm.tqdm(
        range(total_steps, first_step - 1, -save_every), desc="Ckpts"
    )
    cnt = 0
    skipped = 0
    for trained_steps in ckpt_pbar:
      mngr = checkpoint.CheckpointManager(
          ckpt_dir, checkpoint.PyTreeCheckpointer()
      )
      print(
          f"{train_name}; Bsz: {batch_size}, seed: {seed};"
          f" {trained_steps:,d} steps",
          end="; ",
      )
      traj_dir = r[1]["traj_path"]
      traj_file = osp.join(traj_dir, f"pred_traj_step={trained_steps}.hdf5")

      if tf.io.gfile.exists(traj_file):
        print(f"File exists: {traj_file}.")
      else:
        if not tf.io.gfile.exists(osp.join(ckpt_dir, str(trained_steps))):
          print("Skipping! Ckpt file does not exist.")
          skipped += 1
          cnt += 1
          ckpt_pbar.set_postfix({"Skipped": f"{skipped} / {cnt}"})
          continue
        params = mngr.restore(step=trained_steps)["params"]
        if not tf.io.gfile.exists(traj_dir):
          tf.io.gfile.makedirs(traj_dir)
        integrator = choices.Integrator(integrator_choice).dispatch()
        model = choices.Model(model_choice).dispatch(r[1])
        inference_model = functools.partial(
            integrator,
            ode.nn_module_to_dynamics(model),
            params=dict(params=params),
        )
        ics = trajs[:, 0, ...]
        if normalize:
          ics -= mean
          ics /= std
        pt = inference_model(ics, np.arange(all_steps) * dt)
        if normalize:
          pt *= std
          pt += mean
        del params
        print("Generated.", end=" ")
        hdf5_utils.save_array_dict(traj_file, {"pred_traj": pt})
        print(f"Saved to file {traj_file}.")
      cnt += 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  exp_dir = FLAGS.exp_dir
  exps_df = parse_dir(exp_dir)
  dataset_path = exps_df["dataset_path"].unique().tolist()[0]
  trajs, tspan = hdf5_utils.read_arrays_as_tuple(
      dataset_path, ("test/u", "test/t")
  )
  all_steps = trajs.shape[1]
  dt = jnp.mean(jnp.diff(tspan, axis=1))
  print("Num traj:", trajs.shape[0])
  print("traj length (steps):", all_steps)
  print("dt:", dt)
  spatial_downsample = exps_df["spatial_downsample_factor"].tolist()[0]
  if trajs.ndim == 4:
    trajs = trajs[:, :, ::spatial_downsample, :]
  elif trajs.ndim == 5:
    trajs = trajs[:, :, ::spatial_downsample, ::spatial_downsample, :]
  print("Spatial resolution:", trajs.shape[2:-1])

  train_snapshots = hdf5_utils.read_single_array(dataset_path, "train/u")
  mean = jnp.mean(train_snapshots, axis=(0, 1))
  std = jnp.std(train_snapshots, axis=(0, 1))
  del train_snapshots
  print("mean", mean[:10])
  print("std", std[:10])
  generate_pred_traj(exps_df, all_steps, dt, trajs=trajs, mean=mean, std=std)


if __name__ == "__main__":
  app.run(main)
