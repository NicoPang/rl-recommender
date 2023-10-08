import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader

from skorch import NeuralNetRegressor

BATCH_SIZE = 64
NUM_FEATURES = 5
TRAIN_RATIO = 0.8
LEARNING_RATE = 0.0001
EPOCHS = 20

class MF(nn.Module):
    def __init__(self, n_users, n_items, K):
        super().__init__()
        self.user_m = nn.Embedding(n_users, K, dtype = torch.float64) # can include option sparse = True for memory
        self.item_m = nn.Embedding(n_users, K, dtype = torch.float64)
    
    def forward(self, x):
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        user_embeds = self.user_m(user_ids)
        item_embeds = self.item_m(item_ids)
        prod = user_embeds * item_embeds
        
        out = torch.sum(prod, 1)
        
        return out
    
# Matrix factorization with user/item biases
class MF_Bias(MF):
    def __init__(self, n_users, n_items, K, G_b):
        super().__init__(n_users, n_items, K)

        self.user_b = nn.Embedding(n_users, 1, dtype = torch.float64)
        self.item_b = nn.Embedding(n_items, 1, dtype = torch.float64)
        nn.init.zeros_(self.user_b.weight)
        nn.init.zeros_(self.item_b.weight)

        self.G_b = torch.from_numpy(G_b)

    def forward(self, x):
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        out = super().forward(x)

        user_biases = self.user_b(user_ids).squeeze()
        item_biases = self.item_b(item_ids).squeeze()

        out += user_biases + item_biases + self.G_b

        return out


if __name__ == '__main__':
    
    data = np.load('../data/UIR_5core.npz')
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
