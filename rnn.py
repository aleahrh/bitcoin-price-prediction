"""
Bitcoin Price Prediction using RNN
Author: Aleah Hassabo

This project uses a Recurrent Neural Network (RNN) to predict Bitcoin closing prices
from historical market data (Open, High, Low). Achieved R² = 0.9978 on the test set.
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ── Load and preprocess data ──
df = pd.read_csv('coin_Bitcoin.csv')

x = df[['High', 'Low', 'Open']]
y = df[['Close']]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

# 80/20 train/test split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=42)

# Convert to tensors
train_x = torch.tensor(train_x).float().unsqueeze(1)
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float().unsqueeze(1)

# ── Custom Dataset ──
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]

# ── Hyperparameters ──
hidden_size = 128
num_layers = 2
learning_rate = 0.01
batch_size = 40
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# ── Model Architecture ──
class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnnOut, hidden = self.rnn(x)
        rnnOut = rnnOut[:, -1, :]
        rnnOut = self.fc(rnnOut)
        return rnnOut

# ── Training ──
rnn = RNN(input_feature_size=3, hidden_size=hidden_size, num_layers=num_layers).to(device)
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

rnn.train()
for epoch in range(epoch_size):
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out = rnn(inputs)
        loss = criteria(out, targets)
        loss.backward()
        optimizer.step()

print('Finished Training')

# ── Evaluation ──
prediction = []
ground_truth = []

rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs = data[0].to(device)
        targets = data[1]

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out

# Reverse normalization before scoring
prediction = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1))
ground_truth = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1))

r2score = r2_score(prediction, ground_truth)
print(f'R² Score: {r2score:.4f}')
