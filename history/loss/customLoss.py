import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, seq_length, lambda1=1.0, lambda2=1.0, threshold=0.1):
        super(CustomLoss, self).__init__()
        self.seq_length = seq_length
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.threshold = threshold
        self.x = torch.arange(seq_length, dtype=torch.float32)
        self.target_trend = -5/3 * self.x  # 这里假设偏置项 b 为 0

    def forward(self, model_output, ground_truth):
        # 筛选出满足条件（小于阈值）的部分
        mask = model_output < self.threshold
        masked_model_output = model_output[mask]
        masked_target_trend = self.target_trend[mask]

        # 计算趋势损失
        if masked_model_output.numel() > 0:  # 检查是否有满足条件的元素
            trend_loss = F.mse_loss(masked_model_output, masked_target_trend)
        else:
            trend_loss = torch.tensor(0.0, requires_grad=True)  # 如果没有元素满足条件，趋势损失设为0

        # 计算一致性损失
        smoothed_model_output = F.avg_pool1d(model_output.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        smoothed_ground_truth = F.avg_pool1d(ground_truth.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        consistency_loss = F.mse_loss(smoothed_model_output, smoothed_ground_truth)

        # 综合损失
        total_loss = self.lambda1 * trend_loss + self.lambda2 * consistency_loss
        return total_loss

# 示例使用
seq_length = 100
criterion = CustomLoss(seq_length, lambda1=1.0, lambda2=1.0, threshold=0.1)
model_output = torch.randn(seq_length)  # 模型输出示例
ground_truth = torch.randn(seq_length)  # 真实值示例

loss = criterion(model_output, ground_truth)
print("Loss:", loss.item())
