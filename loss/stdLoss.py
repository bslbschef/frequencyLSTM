import torch
import torch.nn as nn


class StdDevLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(StdDevLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        mse_loss = self.mse_loss(outputs, targets)
        std_loss = torch.abs(outputs.std(dim=1) - targets.std(dim=1))
        # std_loss = torch.abs(outputs.std(dim=1) - targets.std(dim=1)).mean()
        return mse_loss + self.alpha * std_loss


criterion = StdDevLoss(alpha=0.1)
