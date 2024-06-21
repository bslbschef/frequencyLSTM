import torch
import torch.nn as nn
import torch.nn.functional as F


class WindSpeedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, flag_bidirectional, time_flag):
        super(WindSpeedGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=flag_bidirectional)
        self.fc = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(hidden_dim*time_flag, hidden_dim*time_flag//2),
            nn.ReLU(),
            nn.Linear(hidden_dim*time_flag//2, hidden_dim*time_flag//4),
            nn.ReLU(),
            nn.Linear(hidden_dim*time_flag//4, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # 使用最后一个时间步的输出进行预测
        # output_gru = self.fc(gru_out[:, -1, :])  # 使用最后一个时间步的输出进行预测
        # 第一个维度：使用ReLU确保输出大于0
        # output = F.relu(output_gru[:, 0:1])
        # output_1 = F.relu(output_gru[:, 0])
        # 第二个维度：直接使用线性输出
        # output_2 = output_gru[:, 1]
        # 第三个维度：使用Tanh并缩放到0-360
        # output_tanh = torch.tanh(output_gru[:, 2])
        # output_3 = (output_tanh + 1) / 2 * 360
        # 合并输出
        # output = torch.stack((output_1, output_2, output_3), dim=1)
        return output
