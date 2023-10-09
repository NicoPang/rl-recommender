import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader

from skorch import NeuralNetRegressor

from model.mf import MF_Bias

BATCH_SIZE = 64
NUM_FEATURES = 5
TRAIN_RATIO = 0.8
LEARNING_RATE = 0.0001
EPOCHS = 20

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
    
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, 1 - TRAIN_RATIO])
    print(f'Training size: {len(train_dataset)}')
    print(f'Testing size: {len(test_dataset)}')
    print('Datasets loaded')
    
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    print('Dataloaders initialized.')
    
    model = MF_Bias(U_size, I_size, NUM_FEATURES, G_b)
    loss_fn = nn.MSELoss
    optimizer = optim.Adam

    regressor = NeuralNetRegressor(
        model,
        max_epochs = EPOCHS,
        criterion = loss_fn,
        optimizer = optimizer,
        optimizer__lr = LEARNING_RATE
    )
    
    regressor.fit(x, y)
