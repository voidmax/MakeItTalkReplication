import torch
import torch.nn as nn
import numpy as np

def create_mask(size, tau):
    iu1 = np.triu_indices(size, 1)
    iu2 = np.triu_indices(size, 1 + tau)
    a = np.ones((size, size)) * (-np.inf)
    a[iu1] = 0
    a[iu2] = -np.inf
    return torch.tensor(a)


class SelfAttentionEncoder(nn.Module):
    def __init__(self, hidden_size_1=256, hidden_size_2=256, hidden_size_4=256, tau=256, nhead=2, dim_feedforward=200, dropout=0.2):
        super().__init__()
        d_model = hidden_size_1 + hidden_size_2
        self.attn = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout,
            batch_first=True, 
        )
        self.linear = nn.Linear(d_model, hidden_size_4)
        self.tau = tau

    def forward(self, x_spk, x_cont):
        t = x_cont.shape[1]
        x = torch.cat([x_cont, x_spk.unsqueeze(1).repeat(1, t, 1)], dim=-1)
        x = self.attn(x)
        #x = x[:, 0, :].unsqueeze(1)
        x = self.linear(x)
        return x

    # def forward(self, x_spk, x_cont):
    #     tau = self.tau
    #     time = x_cont.shape[1]
    #     if time < tau:
    #         tau = time
    #     attn_out = []
    #     for t in range(time):
    #         t_end = min(t + tau, time)
    #         x_cont_window = x_cont[:, t:t_end, :]
    #         x_spk_reshaped = x_spk.unsqueeze(1).repeat(1, t_end - t, 1)
    #         x = torch.cat([x_cont_window, x_spk_reshaped], dim=-1)
    #         x = self.attn(x)[:, 0, :]
    #         attn_out.append(x.unsqueeze(1))
    #     x = torch.cat(attn_out, dim=1)
    #     x = self.linear(x)
    #     return x


class SelfAttentionEncoderPlug(nn.Module):
    def __init__(self, in_hs, out_hs):
        super(SelfAttentionEncoderPlug, self).__init__()
        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, x, y):
        time = y.shape[1]
        x = torch.stack([x] * time, dim=1)
        input = torch.cat([x, y], dim=-1)
        return self.linear(input)
