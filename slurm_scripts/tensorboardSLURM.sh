#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -o /home/projets/RESEARCH/SUR/motion_imitation_ws/tensorboard_out/tb-%J.out

source /home/users/dolivare/.bash_profile #TODO: Your profile
MODEL_DIR=/home/projets/RESEARCH/SUR/motion_imitation_ws/output/zigzag/PPO1_27 #TODO: Your TF model directory
let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

# module load cuda/8.0 #TODO: Your Cuda Module if required

tensorboard --logdir="${MODEL_DIR}" --port=$ipnport