#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -J  tensorboard_server    # name
#SBATCH -o /work/thpaul/tf_tools/tensorflow/im2txt/tb-%J.out #TODO: Where to save your output

# To run as an array job, use the following command:
# sbatch --partition=beards --array=0-0 tensorboardHam.sh
# squeue --user thpaul

source /home/users/dolivare/.bash_profile #TODO: Your profile
MODEL_DIR= $PROJECT_ROOT/motion_imitation_ws/output/PPO1_72/ #TODO: Your TF model directory

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

# module load cuda/8.0 #TODO: Your Cuda Module if required

tensorboard --logdir="${MODEL_DIR}" --port=$ipnport