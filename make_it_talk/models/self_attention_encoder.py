import torch
import torch.nn as nn

class SelfAttentionEncoder(nn.Module):
    def __inti__(self, hidden_size_1, hidden_size_2, hidden_size_4, nhead=2, dim_feedforward=200, dropout=0.2):
        super().__init__()
        d_model = hidden_size_1 + hidden_size_2
        self.attn = nn.SelfAttentionEncoder(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout,
            batch_first=True, 
        )
        self.linear = nn.Linear(d_model, hidden_size_4)

    def forward(self, x_spk, x_cont):
        t = x_cont.shape[1]
        x = torch.cat([x_cont, x_spk.unsqueeze(1).repeat(1, t, 1)], dim=-1)
        x = self.attn(x)
        x = x[:, 0, ...].squeeze(1)
        x = self.linear(x)
        return x


class SelfAttentionEncoderPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(SelfAttentionEncoderPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, x, y):
        time = y.shape[1]
        x = torch.stack([x] * time, dim=1)
        input = torch.cat([x, y], dim=-1)
        return self.linear(input)
