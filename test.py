import gym
import numpy as np

action_upper_bound = []
action_lower_bound = []
action_upper_bound.extend([6.28] * 5)
action_lower_bound.extend([-6.28] * 5)

action_space = gym.spaces.Box(np.array(action_lower_bound), np.array(action_upper_bound), dtype = np.float32)
print(action_space)