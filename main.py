import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, CrossEntropyLoss

from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, global_mean_pool, max_pool, NNConv, Set2Set
import torch_geometric.utils

import matplotlib.pyplot as plt
import networkx as nx

dataset = QM9('/data/QM9')
dataset = dataset.shuffle()
target = 0
dim = 64
batch_size=256
train_ratio = 0.7

# Normalize data
m = dataset.data.y.mean(dim=0, keepdim=True)
s = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - m) / s
m, s = m[:, target].item(), s[:, target].item()

dataset = dataset[0:(int) ( len(dataset) * 0.1)]
train_test_position = (int) ((1-train_ratio) * len(dataset))

# train split into data loader
test = dataset[:train_test_position - 1000]
test_loader = DataLoader(test, batch_size=batch_size)

# test split into data loader
t = dataset[train_test_position:]
train_loader = DataLoader(t, batch_size=batch_size)

# validation data
val = dataset[train_test_position - 1000: train_test_position]
val_loader = DataLoader(val, batch_size=128)

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(4, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.linear1 = torch.nn.Linear(2 * dim, dim)
        self.linear2 = torch.nn.Linear(dim, 19)

    def forward(self, data):
        out = F.relu(self.linear0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out.view(-1)
# Using Google T4 GPU, so Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
# Adam optimizer with learning rate of 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.0001)
print(model)
def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Cross Entropy loss
        loss = F.mse_loss(model(data), data.y.view(-1))
        # BACKPROPAGATION!!!
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)
def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += abs(model(data) * s - data.y.view(-1,1) * s).sum().item()
    return error / len(loader.dataset)
for epoch in range(1, 101):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print(f'Epoch: {epoch:03d}, LR: {lr:.7f} Loss: {loss:.7f}, '
          f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')
# Graph of loss over time
epochs = []
loss = []
for i in range(len(metrics)):
  epochs.append(metrics[i]['Epoch'])
  loss.append(metrics[i]['Loss'])
plt.plot(epochs, loss)
print(dataset[-1].y)
print(model(dataset[-1].to(device)))
