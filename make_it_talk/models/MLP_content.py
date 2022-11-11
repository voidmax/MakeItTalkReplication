import torch
import torch.nn as nn

class MLPContent(nn.Module):
    def __init__(self, hidden_size_4, landmarks_dim, dropout=0.5) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_size_4),
            nn.Linear(hidden_size_4, hidden_size_4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size_4),
            nn.Linear(hidden_size_4, landmarks_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        return self.mlp(input)

class MLPContentPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(MLPContentPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, emb, land):
        time = emb.shape[1]
        land = torch.stack([land] * time, dim=1)
        x = torch.cat([emb, land], dim=-1)
        return self.linear(x)