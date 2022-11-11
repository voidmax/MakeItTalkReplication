import torch
import torch.nn as nn

class MLPSpeakerEmbedding(nn.Module):
    def __init__(self, hidden_size_2, hidden_size_3) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size_2, hidden_size_3)

    def forward(self, input):
        return self.linear(input)
            
class MLPSpeakerEmbeddingPlug(nn.Module):
    def __init__(self, hidden_size_2, hidden_size_3) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size_2, hidden_size_3)

    def forward(self, input):
        return self.linear(input)