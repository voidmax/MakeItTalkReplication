import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return torch.transpose(x, 1, 2)

class MLPContent(nn.Module):
    def __init__(self, hidden_size_4=256, landmarks_dim=68*3, dropout=0.5) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            Transpose(),
            nn.BatchNorm1d(hidden_size_4 + landmarks_dim),
            Transpose(),
            nn.Linear(hidden_size_4 + landmarks_dim, hidden_size_4),
            nn.ReLU(),
            nn.Dropout(dropout),
            Transpose(),
            nn.BatchNorm1d(hidden_size_4),
            Transpose(),
            nn.Linear(hidden_size_4, landmarks_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_content, input_landmarks):
        # print("MLP_CONT: ", input_content.shape, input_landmarks.shape)
        time = input_content.shape[1]
        landmarks = input_landmarks.unsqueeze(1).repeat(1, time, 1)
        x = torch.cat([input_content, landmarks], dim=-1)
        # print("MLP_CONT_x: ", x.shape)
        return self.mlp(x)

class MLPContentPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(MLPContentPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, emb, land):
        time = emb.shape[1]
        land = torch.stack([land] * time, dim=1)
        x = torch.cat([emb, land], dim=-1)
        return self.linear(x)