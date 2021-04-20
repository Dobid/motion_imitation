import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

env = gym.make('motion_imitation:A1GymEnv-v1', render=True)
# env = gym.make('LunarLanderContinuous-v2')
# print("ac_space gym = ", env.action_space.low, " ", env.action_space.high)
model = PPO1(MlpPolicy, env, verbose=1)
print("------------ TRAINING STARTED -----------------")
model.learn(total_timesteps=25000)
model.save("ppo1_imitation")

del model
model = PPO1.load("hello")
obs = env.reset()

while True:
    action, _states, = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()