import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """Paper: two stacked LSTM layers (128, 64) + dense. Default here is scaled down for CPU."""
    def __init__(self, n_features, h1=64, h2=32):
        super().__init__()
        self.l1 = nn.LSTM(n_features, h1, batch_first=True); self.l2 = nn.LSTM(h1, h2, batch_first=True)
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        h, _ = self.l1(x); h, _ = self.l2(h); return self.out(h[:, -1]).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, n_features, filters=32, h1=64, h2=32):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(n_features, filters, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = LSTMNet(filters, h1, h2)

    def forward(self, x):
        return self.lstm(self.cnn(x.transpose(1, 2)).transpose(1, 2))
