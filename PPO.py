import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym
from gym import spaces
import copy
import torch.optim as optim
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
input_feature_dim = 3  # Each input element is a 1x3 vector
embed_size = 128
target_dim = 3
block_size = 100
num_heads = 32
max_iters = 1200
batch_size = 32
eval_iters = 200
eval_interval = 10
num_layers=12

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        K = self.keys(x)
        Q = self.queries(x)
        V = self.values(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.embed_size ** 0.5
        attention = torch.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention, V)
        return attended

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert embed_size % num_heads == 0

        self.head_dim = embed_size // num_heads

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        keys = self.keys(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = self.queries(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.values(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        attention_scores = torch.einsum("bnqh,bnkh->bnqk", [queries, keys]) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention_scores, dim=-1)

        attended = torch.einsum("bnqk,bnkv->bnqv", [attention, values]).reshape(batch_size, seq_length, self.embed_size)

        output = self.fc_out(attended)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.dropout1 = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, value):
        x = self.norm1(value)
        attention_output = self.attention(x)
        x = value + self.dropout1(attention_output)  # Residual connection and dropout after attention
        x = self.norm2(x)
        feed_forward_output = self.feed_forward(x)
        out = value + self.dropout2(feed_forward_output)  # Residual connection and dropout after FFN
        return out

# Positional Encoding in Encoder class should be moved to the device
class Encoder(nn.Module):
    def __init__(self, input_feature_dim, embed_size, num_heads, num_layers, seq_length):
        super(Encoder, self).__init__()
        self.input_fc = nn.Linear(input_feature_dim, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, embed_size)).to(device)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_fc(x)) + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        return x
    
    def to_cpu(self):
        # Move the entire model to CPU
        self.input_fc.to('cpu')
        self.positional_encoding.data = self.positional_encoding.data.cpu()
        for layer in self.layers:
            layer.to('cpu')
        self.relu.to('cpu')
        torch.cuda.empty_cache()

class EncoderDecoderModelWithMultiHeadAttention(nn.Module):
    def __init__(self, input_feature_dim, embed_size, target_dim, seq_length, num_heads, num_layers):
        super(EncoderDecoderModelWithMultiHeadAttention, self).__init__()
        self.encoder = Encoder(input_feature_dim, embed_size, num_heads, num_layers, seq_length)
        self.decoder = nn.Sequential(
            nn.Linear(embed_size, target_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_pooled = torch.mean(encoded, dim=1)
        decoded = self.decoder(encoded_pooled)

        return decoded

    def to_cpu(self):
        self.encoder.to_cpu()
        for layer in self.decoder:
            layer.to('cpu')
        torch.cuda.empty_cache()
import time 
def start_time():
    return time.time()

def elapsed(a):
    return time.time()-a
model_path = "C:/Users/yueze/Desktop/trained_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelt = EncoderDecoderModelWithMultiHeadAttention(input_feature_dim, embed_size, target_dim, block_size+1, num_heads, num_layers)
modelt.load_state_dict(torch.load(model_path, map_location=device))
modelt.to(device)
print("Model loaded from", model_path)
class DummyHardware:
    """Simulates hardware behavior for the environment."""
    @staticmethod
    def implement(actions,t, measurement_1350, measurement_1550):
        # Apply action effects with cosine function
        measurement_1350 += np.cos(actions)+ np.sin(t)
        # Wrap around if the value exceeds 2.2 or goes below -2.2
        
        while np.any(measurement_1350 > 2.2) or np.any(measurement_1350 < -2.2):
            measurement_1350 = np.where(measurement_1350 > 2.2, measurement_1350 - 4.4, measurement_1350)
            measurement_1350 = np.where(measurement_1350 < -2.2, measurement_1350 + 4.4, measurement_1350)
        
        # Apply action to 1550 nm state with a multiplier of 1
        measurement_1550 += np.cos(3 * actions) + np.sin(3 * t)
        
        while np.any(measurement_1550 > 2.2) or np.any(measurement_1550 < -2.2):
            measurement_1550 = np.where(measurement_1550 > 2.2, measurement_1550 - 4.4, measurement_1550)
            measurement_1550 = np.where(measurement_1550 < -2.2, measurement_1550 + 4.4, measurement_1550) 
        return measurement_1350, measurement_1550

    @staticmethod
    def drift(t, measurement_1350, measurement_1550):
        # Apply action effects with cosine function
        measurement_1350 += np.sin(t)
        while np.any(measurement_1350 > 2.2) or np.any(measurement_1350 < -2.2):
            measurement_1350 = np.where(measurement_1350 > 2.2, measurement_1350 - 4.4, measurement_1350)
            measurement_1350 = np.where(measurement_1350 < -2.2, measurement_1350 + 4.4, measurement_1350)
        measurement_1550 += np.sin(3 * t)
        while np.any(measurement_1550 > 2.2) or np.any(measurement_1550 < -2.2):
            measurement_1550 = np.where(measurement_1550 > 2.2, measurement_1550 - 4.4, measurement_1550)
            measurement_1550 = np.where(measurement_1550 < -2.2, measurement_1550 + 4.4, measurement_1550)
   
        return measurement_1350, measurement_1550

    @staticmethod
    def measure(wavelength, t, initial_state_1550=None):
        # Simulate a hardware measurement with random values within a range
        if wavelength == 1350:
            return env.optimal_state + np.sin(t)
        elif wavelength == 1550 and initial_state_1550 is not None:
            return initial_state_1550 + np.sin(3 * t)
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model with specific input, hidden, and output sizes
input_size = 306  # Example: 102 measurements * 3 features
hidden_size = 128
output_size = 3  # Assuming the output is a 3-dimensional vector


class MyCustomEnv(gym.Env):
    def __init__(self, modelt, device):
        super(MyCustomEnv, self).__init__()
        
        self.modelt = modelt
        self.device = device
        
        self.optimal_state = np.array([0, 1, 2.2])
        self.df_1550 = []
        self.t = 0
        
        # Action space: assuming a 3-dimensional continuous action space
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        
        # Observation space: for simplicity, let's assume it's a 100x3 array concatenated with two additional vectors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 306), dtype=np.float32)
        
        # Variables for after the action executes
        self.measured_1550 = None
        self.measured_1350= None
        self.drifted_1550= None
        self.drifted_1350=None
        
    def reset(self):
        self.t = 0
        action = np.array([0,0,0])
        self.measured_1350 = np.array(DummyHardware.measure(1350, self.t))
        self.data_to_correct_for = self.optimal_state - self.measured_1350
        action = self.data_to_correct_for
        self.measured_1550 = np.random.uniform(-2, 2, 3)
        
        self.df_1550 = []
        while len(self.df_1550) < 100:
            measured_1350 = copy.deepcopy(self.measured_1350)
            measured_1550 = copy.deepcopy(self.measured_1550)
            self.t += 1
            self.measured_1350, self.measured_1550 = DummyHardware.implement(action, self.t, measured_1350, measured_1550)
            self.df_1550.append(copy.deepcopy(self.measured_1550))
        if len(self.df_1550) > 100:
            self.df_1550.pop(0)
        
        stacked_df = np.stack(self.df_1550)
        optimal_state_reshaped = self.optimal_state.reshape(1, -1)
        stacked_df_with_optimal = np.concatenate((stacked_df, optimal_state_reshaped), axis=0)
        tensor_df = torch.tensor(stacked_df_with_optimal, dtype=torch.float32).to(self.device)
        output= self.modelt(tensor_df)
        next_state_measurements = np.array(self.df_1550)
        output_numpy = output.cpu().detach().numpy().reshape(1, -1)
        state = np.concatenate((next_state_measurements, output_numpy, optimal_state_reshaped), axis=0).reshape(1, -1)
        return state
    
    def step(self, action):
        measured_1350 = copy.deepcopy(self.measured_1350)
        measured_1550 = copy.deepcopy(self.measured_1550)
        self.t += 1
        self.measured_1350, self.measured_1550 = DummyHardware.implement(action, self.t, measured_1350, measured_1550)
        self.df_1550.append(copy.deepcopy(self.measured_1550))
        if len(self.df_1550) > 100:
            self.df_1550.pop(0)
        
        stacked_df = np.stack(self.df_1550)
        optimal_state_reshaped = self.optimal_state.reshape(1, -1)
        stacked_df_with_optimal = np.concatenate((stacked_df, optimal_state_reshaped), axis=0)
        tensor_df = torch.tensor(stacked_df_with_optimal, dtype=torch.float32).to(self.device)
        output= self.modelt(tensor_df)
        next_state_measurements = np.array(self.df_1550)
        output_numpy = output.cpu().detach().numpy().reshape(1, -1)
        next_state = np.concatenate((next_state_measurements, output_numpy, optimal_state_reshaped), axis=0).reshape(1, -1)
        
        reward = -np.mean((self.measured_1350 - self.optimal_state) ** 2)
        
        done = False  # You can implement logic for termination here if needed
        info = {}  # Additional information for debugging
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        print(f"t = {self.t}, Measured 1350: {self.measured_1350}, Measured 1550: {self.measured_1550}")
    
    def close(self):
        pass
env = MyCustomEnv(modelt=modelt, device=device)

# Initialize the RecurrentPPO model
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    lstm_hidden_size=256,
)

# Create and train the RecurrentPPO model
model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10000)
model.save("recurrent_ppo_custom_env")

# Load and continue training or evaluation
model = RecurrentPPO.load("recurrent_ppo_custom_env", env=env)

# Define a callback for logging rewards
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"]
        self.rewards.append(reward)
        return True

# Create the callback for logging rewards
callback = RewardLoggerCallback()

# Train the model with logging
model.learn(total_timesteps=10000, callback=callback)

# Save rewards for further analysis
rewards = callback.rewards

# Visualize the Training Progress
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('PPO Agent Improvement Over Time')
plt.show()