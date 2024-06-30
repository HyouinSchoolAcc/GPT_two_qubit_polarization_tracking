#import some stuff
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import random
import math
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import ipywidgets as widgets
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_dir = "C:/Users/凤凰院凶真/Desktop/GPT_two_qubit_polarization_tracking/20240506_2131.csv"
df = pd.read_csv(data_dir)
print(df.head)

features = ["AI0","AI1","AI2"]
targets  = ["AI3","AI4","AI5"]
x = df[features]  # Dropping original targets as we'll use aligned targets
y = df[targets]  # Using aligned targets
split_idx = int(len(df) * 0.9)
length_read=len(df)
# Split into training and testing sets
x_train = x.iloc[:split_idx]
y_train = y.iloc[:split_idx]
x_test = x.iloc[split_idx:]
y_test = y.iloc[split_idx:]

input_feature_dim = 3  # Each input element is a 1x3 vector
embed_size = 64
target_dim = 3
block_size = 40
num_heads = 16
max_iters = 144
batch_size = 10
eval_iters = 200
eval_interval = 1
num_layers=5

def get_batch3(split):
    # Select the correct data split
    if split == 'train':
        a, b, max_index = x_train, y_train, int(length_read * 0.9) - block_size - 1
    else:  # split == 'test'
        a, b, max_index = x_test, y_test, length_read - (int(length_read * 0.9) + block_size + 1)

    # Generate random indices for batch selection, ensuring they're within bounds
    ix = torch.randint(0, max_index, (batch_size,))

    # Initialize lists to hold the batches
    x_batch = []
    y_batch = []

    for i in ix:
        try:
                        # Extract the sequence from 'a' and the corresponding target from 'b'                                                           
            seq = torch.tensor(a.iloc[i.item():i.item() + block_size].astype(np.float32).values, dtype=torch.float32).to(device)
            target = torch.tensor(b.iloc[i.item() + block_size].astype(np.float32).values, dtype=torch.float32).to(device)

            x_batch.append(seq)
            y_batch.append(target)
        except IndexError as e:
            print(f"IndexError for index {i.item()}: {str(e)}")
            print(f"Attempting to access index [{i.item()}:{i.item() + block_size}] in 'a' with shape {a.shape}")
            print(f"Attempting to access index {i.item() + block_size} in 'b' with shape {b.shape}")
            # Optionally, break or continue depending on desired behavior on error
            break  # or continue

    if not x_batch or not y_batch:
        print("Error: Batch could not be created due to index issues.")
        return None, None

    # Stack the collected sequences and targets into tensors
    xstack = torch.stack(x_batch)
    ystack = torch.stack(y_batch)

    return xstack, ystack
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch3(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

class EncoderDecoderModelWithMultiHeadAttention(nn.Module):
    def __init__(self, input_feature_dim, embed_size, target_dim, seq_length, num_heads, num_layers):
        super(EncoderDecoderModelWithMultiHeadAttention, self).__init__()
        self.encoder = Encoder(input_feature_dim, embed_size, num_heads, num_layers, seq_length)
        self.decoder = nn.Sequential(
            nn.Linear(embed_size, target_dim),
        )

    def forward(self, x, targets):
        encoded = self.encoder(x)
        encoded_pooled = torch.mean(encoded, dim=1)
        decoded = self.decoder(encoded_pooled)

        loss = criterion(decoded, targets)

        return decoded, loss
    


actuals = []
predictions = []
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EncoderDecoderModelWithMultiHeadAttention(input_feature_dim, embed_size, target_dim, block_size, num_heads, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for iter in range(max_iters):
        # Evaluate the loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch3('train')
        xb, yb = xb.to(device), yb.to(device)  # Ensure these tensors are on the correct device

        predictions, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("Loss:", loss.item())