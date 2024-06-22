import torch
import torch.nn as nn
import torch.nn.functional as F


class WindSpeedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, flag_bidirectional, time_flag):
        super(WindSpeedGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=flag_bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*time_flag, hidden_dim*time_flag//2),
            nn.ReLU(),
            nn.Linear(hidden_dim*time_flag//2, hidden_dim*time_flag//4),
            nn.ReLU(),
            nn.Linear(hidden_dim*time_flag//4, output_dim),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, :])
        return output
