import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return torch.transpose(x, 1, 2)

class MLPSpeaker(nn.Module):
    def __init__(self, hidden_size_4=256, landmarks_dim=68*3, dropout=0.5) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_size_4 + landmarks_dim, 512),
            Transpose(),
            nn.BatchNorm1d(512),
            Transpose(),
            nn.ReLU(),
            nn.Linear(512, 256),
            Transpose(),
            nn.BatchNorm1d(hidden_size_4),
            Transpose(),
            nn.ReLU(),
            nn.Linear(256, landmarks_dim),
        )

    def forward(self, input_audio, input_landmarks):
        print("MLP_CONT: ", input_audio.shape, input_landmarks.shape)
        time = input_audio.shape[1]
        landmarks = input_landmarks.unsqueeze(1).repeat(1, time, 1)
        x = torch.cat([input_audio, landmarks], dim=-1)
        # print("MLP_CONT_x: ", x)
        return self.fc(x)


class MLPSpeakerPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(MLPSpeakerPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, emb, land):
        time = emb.shape[1]
        land = torch.stack([land] * time, dim=1)
        x = torch.cat([emb, land], dim=-1)
        return self.linear(x)


