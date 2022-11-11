import torch
import torch.nn as nn

class MLPSpeaker(nn.Module):
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

