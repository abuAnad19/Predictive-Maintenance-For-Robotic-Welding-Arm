import torch
import torch.nn as nn

class SensorLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]         # take last time-step
        x = self.drop(x)
        x = self.fc(x)
        return x                 # logits
