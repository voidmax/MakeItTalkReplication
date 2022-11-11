import torch
import torch.nn as nn
import face_alignment
import numpy as np

class FacialLandmarksExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,  device='cpu')

    def forward(self, input):
        return torch.stack([
            torch.Tensor(np.array(self.model.get_landmarks(i))) for i in input
        ]).view(input.shape[0], -1)
            
