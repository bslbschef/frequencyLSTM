import torch
import torch.nn as nn
import torch.nn.functional as F


class WindSpeedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, flag_bidirectional):
        super(WindSpeedGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=flag_bidirectional)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, :, :])
        return output
