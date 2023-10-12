import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, TensorDataset, DataLoader

from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from model.mf import MF_Bias
from model.utility import RMSELoss
from train.saving import save_model_results

BATCH_SIZE = 64
NUM_FEATURES = 10
LEARNING_RATE = 0.0001
EPOCHS = 25
DECAY = 1e-3

if __name__ == '__main__':
    
    data = np.load('../datasets/processed/UIR_5core.npz')
    x = data['x']
    y = data['y']
    U_size = data['U_size']
    I_size = data['I_size']
    G_b = data['G_b']
    
    print(len(x))
    print(len(y))
    print(U_size)
    print(I_size)
    print(G_b)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    y_train, y_test = torch.tensor(y_train, dtype = torch.float64), torch.tensor(y_test, dtype = torch.float64)
    
    model = MF_Bias(U_size, I_size, NUM_FEATURES, G_b)
    loss_fn = RMSELoss()
    optimizer = Adam

    regressor = NeuralNetRegressor(
        model,
        criterion = loss_fn,
        optimizer = optimizer,
        optimizer__param_groups = [
            ('user_m.weight', {'weight_decay': DECAY}),
            ('item_m.weight', {'weight_decay': DECAY})
        ],
        optimizer__lr = LEARNING_RATE,
        batch_size = BATCH_SIZE,
        max_epochs = EPOCHS
    )

    print(list(regressor.get_all_learnable_params()))
    
    save_model_results(regressor, x, y, f'../results/bs{BATCH_SIZE}K{NUM_FEATURES}LR{LEARNING_RATE}E{EPOCHS}D{DECAY}.npz')