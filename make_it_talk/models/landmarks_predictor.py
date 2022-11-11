import torch
import torch.nn as nn

class LandmarksPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, base_lancmarks, content_delta, speaker_delta):
        return base_lancmarks + content_delta + speaker_delta
            
