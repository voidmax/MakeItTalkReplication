import torch
import torch.nn as nn

class SelfAttentionEncoderPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(SelfAttentionEncoderPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, x, y):
        time = y.shape[1]
        x = torch.stack([x] * time, dim=1)
        input = torch.cat([x, y], dim=-1)
        return self.linear(input)
