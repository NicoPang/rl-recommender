from test import MF_Bias, BATCH_SIZE, TRAIN_RATIO, NUM_FEATURES, LEARNING_RATE, EPOCHS
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader
from skorch import NeuralNetRegressor

from sklearn.model_selection import GridSearchCV, ParameterGrid
from skorch.callbacks import Callback
import pandas as pd

EPOCHS = 200
CV = 2

# Grid search for the number of featurers
if __name__ == "__main__":
    data = np.load(
        os.path.abspath("../rl-recommender/datasets/processed/UIR_5core.npz")
    )
    x = data["x"]
    y = data["y"]
    U_size = data["U_size"]
    I_size = data["I_size"]
    G_b = data["G_b"]
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, 1 - TRAIN_RATIO])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # New -----------------------------------------------------------------------------
    '''
    class RecordResultsCallback(Callback):
        def __init__(self, results=list()):
            super().__init__()
            self.results = results

        def on_epoch_begin(self, net, dataset_train, dataset_valid):
            model_K = net.module_.K
            print(f"Training for K = {model_K}")
            # Clear lists at the start of a new value of K
            self.train_lost = []
            self.val_loss = []
            self.num_epochs = []

        def on_epoch_end(self, net, dataset_train, dataset_valid):
            self.train_lost.append(net.history[-1, "train_loss"])
            self.val_loss.append(net.history[-1, "valid_loss"])
            self.num_epochs.append(len(net.history))
            
        def on_train_end(self, net, X=None, y=None, **kwargs):
            df = pd.DataFrame({'K': [net.module_.K], 'Train Loss': self.train_lost, 'Validation Loss': self.val_loss, 'Epochs': self.num_epochs})
            self.results.append(df)


    callback_results = list()

    model = MF_Bias(n_users=U_size, n_items=I_size, K=NUM_FEATURES, G_b=G_b)
    regressor = NeuralNetRegressor( #regressor.callbacks_
        module=model,
        module__n_users=U_size,
        module__n_items=I_size,
        optimizer=optim.Adam,
        optimizer__lr=LEARNING_RATE,
        criterion=nn.MSELoss,
        max_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[RecordResultsCallback()], #I REMOVED THE callback_result
        history=True,
    )

    param_grid = {
        "module__n_users": [U_size],
        "module__n_items": [I_size],
        "module__K": [i for i in range(FEATURE_START, FEATURE_END) if i % 2 == 0],
        "module__G_b": [G_b],
    }
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=CV,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(x, y)
    '''
    #------ Version 2
    class RecordResultsCallback(Callback):
        def __init__(self, results=list()):
            super().__init__()
            self.results = results
            
        def on_train_begin(self, net, X=None, y=None, **kwargs):
            self.train_lost = []
            self.val_loss = []
            self.num_epochs = []
            self.model_K = net.module_.K
            print(f"Training for K = {self.model_K}")

        def on_epoch_end(self, net, dataset_train, dataset_valid):
            self.train_lost.append(net.history[-1, "train_loss"])
            self.val_loss.append(net.history[-1, "valid_loss"])
            self.num_epochs.append(len(self.train_lost))

        def on_train_end(self, net, X=None, y=None, **kwargs):         
            temp = {'Epochs': self.num_epochs, 'Train Loss': self.train_lost, 'Validation Loss': self.val_loss,'K': [self.model_K]}
            df = pd.DataFrame.from_dict(temp, orient='index')
            self.results = df.transpose()

    param_grid = {
        "module__n_users": [U_size],
        "module__n_items": [I_size],
        "module__K": [2,4,5,8,16,32,50],
        "module__G_b": [G_b],
        }

    all_results = list()

    for params in ParameterGrid(param_grid):
        model = MF_Bias(
            n_users=params['module__n_users'],
            n_items=params['module__n_items'],
            K=params['module__K'],
            G_b=params['module__G_b']
        )

        regressor = NeuralNetRegressor(
            module=model,
            max_epochs=EPOCHS,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            optimizer__lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            callbacks=[RecordResultsCallback()],
            )
            
        regressor.fit(x, y)
        
        callback_results = regressor.callbacks[0].results
        all_results.append(callback_results)
    
    print(all_results)
    
    with pd.ExcelWriter('../rl-recommender/src/param_search.xls', engine='xlsxwriter') as writer:
        for i, df in enumerate(all_results):
            sheet_name = f'Sheet_{i}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    