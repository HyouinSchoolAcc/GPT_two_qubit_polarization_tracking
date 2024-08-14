import gym
import numpy as np
import matplotlib.pyplot as plt
from baselines.common.policies import build_lstm_policy
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv
from baselines import logger

# Step 1: Create the environment
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# Step 2: Define the LSTM policy
policy = build_lstm_policy(env, lstm_hidden_size=64)

# Step 3: Custom Callback to Log Rewards
class RewardLoggerCallback:
    def __init__(self):
        self.rewards = []

    def __call__(self, locals_, globals_):
        # Get the current rewards
        reward = locals_["rewards"]
        self.rewards.append(np.mean(reward))
        return True

# Step 4: Set up the logger and callback
logger.configure()
reward_logger = RewardLoggerCallback()

# Step 5: Train the PPO agent with LSTM policy
model = ppo2.learn(
    env=env,
    network=policy,
    total_timesteps=10000,
    nsteps=128,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    callback=reward_logger
)

# Step 6: Visualize the Training Progress
plt.plot(reward_logger.rewards)
plt.xlabel('Training Steps')
plt.ylabel('Mean Reward')
plt.title('PPO Agent Improvement Over Time')
plt.show()
