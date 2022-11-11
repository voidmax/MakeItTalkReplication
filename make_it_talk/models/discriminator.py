import torch
import torch.nn as nn

class DiscriminatorPlug(nn.Module):
    def __init__(self, in_hs1, in_hs2, in_hs3):
        super(DiscriminatorPlug, self).__init__()
        self.linear = nn.Linear(in_hs1 + in_hs2 + in_hs3, 1)

    def forward(self, x, y, z):
        # predicted_landmarks, personal_processed (after lstm_personal), speaker_processed (after mlp_speaker)
        time = x.shape[1]
        batch = x.shape[0]
        z = torch.stack([z] * time, dim=1)
        input = torch.cat([x,y,z], dim=-1)
        return self.linear(input).reshape(batch, time)
