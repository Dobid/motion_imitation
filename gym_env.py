import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

# env = gym.make('motion_imitation:A1GymEnv-v1', render=True)
env = gym.make('LunarLanderContinuous-v2')
model = PPO1(MlpPolicy, env, verbose=1)
print(env.action_space)
print("------------ TRAINING STARTED -----------------")
model.learn(total_timesteps=25000)
model.save("ppo1_imitation")

del model
model = PPO1.load("/home/sur/Documents/full_trained.zip")
obs = env.reset()

while True:
    action, _states, = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='rgb_array')
env.close()