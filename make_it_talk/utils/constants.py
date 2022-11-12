import torch

landmark_classes = [
    torch.arange(0, 17),  # head
    torch.arange(17, 22),  # left eyebrow
    torch.arange(22, 27),  # right eyebrow
    torch.arange(27, 36),  # nose
    torch.arange(36, 42),  # left eye
    torch.arange(42, 48),  # right eye
    torch.arange(48, 60),  # ? outer edge of the mouth ?
    torch.arange(60, 68),  # ? inner edge of the mouth ?
]
