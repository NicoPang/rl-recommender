import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error

class RMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.epsilon = epsilon
    
    def forward(self, pred, true):
        loss = torch.sqrt(self.mse(pred, true) + self.epsilon)
        return loss
    
def RMSE(net, X, y):
        mse = nn.MSELoss()
        loss = np.sqrt(mean_squared_error(net.predict(X), y) + 1e-8)
        return loss