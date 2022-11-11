import torch
import torch.nn as nn

class LSTMSpeechContent(nn.Module):
    def __init__(self, audio_dim, hidden_size_1, dropout=0.5) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=audio_dim, 
                                hidden_size=hidden_size_1,
                                num_layers=1,
                                dropout=dropout,
                                bidirectional=False,
                                batch_first=True)

    def forward(self, input):
        out, _ = self.lstm(input)
        return out
            
class LSTMSpeechContentPlug(nn.Module):
    def __init__(self, in_hs, out_hs) -> None:
        super().__init__()

        self.linear = nn.Linear(in_hs, out_hs)

    def forward(self, input):
        return self.linear(input)