import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.epsilon = epsilon

    def forward(self, pred, true):
        loss = torch.sqrt(self.mse(pred, true) + self.epsilon)
        return loss
