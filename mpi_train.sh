mpiexec -n 7 python3 motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/turn.txt --int_save_freq 10000000 --sync_reference --total_timesteps 2000000000
