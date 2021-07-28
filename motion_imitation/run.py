# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from operator import pos
import os
import inspect
from motion_imitation.utilities.debug_logger import logd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 32

ENABLE_ENV_RANDOMIZER = True

def set_rand_seed(seed=None):
  if seed is None:
    seed = int(time.time())

  seed += 97 * MPI.COMM_WORLD.Get_rank()

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant', #Change to linear
               policy_kwargs=policy_kwargs,
               tensorboard_log=output_dir,
               verbose=1)
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
  if (output_dir == ""):
    save_path = None
  else:
    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  

  callbacks = []
  # Save a checkpoint every n steps
  if (output_dir != ""):
    if (int_save_freq > 0):
      int_dir = os.path.join(output_dir, "intermedate")
      callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                          name_prefix='model'))
  print("TOTAL TIMESTEPS = ", total_timesteps)
  model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)

  return

def test(model, env, num_procs, sync_ref, num_episodes=None):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o = env.reset()
  # cmd = []
  while episode_count < num_local_episodes:
    # input("press enter for steping the env")
    a, _ = model.predict(o, deterministic=True)
    o, r, done, info = env.step(a)
    # print("cmd_vel = ", o[-6:])
    # cmd.append(o[-6:])
    curr_return += r

    if done:
      # cmd = np.array(cmd)
      # robot_positions = np.array(env._robot_positions)
      # ref_positions = np.array(env._ref_positions)
      # X = robot_positions[:,0]
      # X = np.linspace(0, cmd.shape[0], cmd.shape[0])
      # cmd = cmd[:,-3:]
      # y_pos = np.column_stack((robot_positions[:,1], ref_positions[:,1]))
      # plot_graphs(X,cmd)
      if sync_ref:
        o = env.reset()
      sum_return += curr_return
      episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def plot_graphs(X,cmd):
  cmd = np.transpose(cmd)
  # X = np.transpose(X)
  labels = ['roll', 'pitch', 'yaw']
  print(cmd.shape)
  for y, label in zip(cmd, labels):
    plt.plot(X, y, label=label)
  # plt.ylim(-1,1)
  plt.legend()
  plt.show()

# def plot_graphs(cmd):
#   # X = np.linspace(0,1800,1800)
#   cmd = np.transpose(cmd)
#   # fig, (ax1, ax2) = plt.subplots(2)
#   X = np.linspace(0,cmd.shape[1],cmd.shape[1])
#   labels = ['x_lin_vel', 'y_lin_vel', 'z_lin_vel', 'roll_vel', 'pitch_vel', 'yaw_vel']

#   # y_lin_vel = cmd[1]
#   # avg_y_vel = np.mean(y_lin_vel)
#   # ax1.plot(X, y_lin_vel)
#   # y_lin_vel = y_lin_vel - avg_y_vel

#   # pos_y = [0]
#   # for vel in y_lin_vel:
#   #   pos_y.append(pos_y[-1]+vel*0.033)
#   # del pos_y[0]
#   # ax2.plot(X, pos_y)

#   for y, label in zip(cmd, labels):
#     plt.plot(X, y, label=label)
#   plt.legend()
#   plt.show()

  # print("AVG_Y_VEL = ", avg_y_vel)

def main():
  # Parsing arguments from the python command
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None) # custom random seed
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train") # choose mode : train or test
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt") # motion file, motion capture reference path
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False) # set visualization, disable for training
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output") # output directory path for storing tensorboard files and model.zip
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None) # number of episodes for 1 rollout during 
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="") # model file to load for testing or retraining
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8) # total number of timesteps for training a policy
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps
  arg_parser.add_argument("--sync_reference", dest="sync_reference", action="store_true", default=False) # sync reference ghost to robot body

  args = arg_parser.parse_args()
  
  # get number of CPUs to use from the mpiexec cmd
  num_procs = MPI.COMM_WORLD.Get_size()
  
  # doesn't use GPU computing
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

  # enable environnement (simulation) randomizer
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

  # building custom gym env
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        arg_enable_cycle_sync=args.sync_reference,
                                        enable_rendering=args.visualize)

  if args.model_file != "": # loading model from file (for test or re-training)
    print("loading model from provided model file")
    model = ppo_imitation.PPOImitation.load(load_path=args.model_file, env=env)
    model.tensorboard_log = args.output_dir
  else: # creating new model (for training from scratch)
    print("no provided model file, creating a new one")
    model = build_model(env=env,
                      num_procs=num_procs,
                      timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                      optim_batchsize=OPTIM_BATCHSIZE,
                      output_dir=args.output_dir)

  if args.mode == "train": # train mode
      train(model=model, 
            env=env, 
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq)
  elif args.mode == "test": # test mode
      test(model=model,
           env=env,
           num_procs=num_procs,
           num_episodes=args.num_test_episodes,
           sync_ref=args.sync_reference)
  else:
      assert False, "Unsupported mode: " + args.mode
  # Use this in the terminal to start the tensorboard server  
  # tensorboard --logdir=./tfb_logs/ --port=8090 --host=127.0.0.1
  return

if __name__ == '__main__':
  main()
