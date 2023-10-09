from test import MF_Bias, BATCH_SIZE, TRAIN_RATIO, NUM_FEATURES, LEARNING_RATE, EPOCHS
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader
from skorch import NeuralNetRegressor

from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Callback
import pandas as pd

EPOCHS = 1

CV = 2
FEATURE_START = 2
FEATURE_END = 5

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
    class RecordResultsCallback(Callback):
        def __init__(self, results_list):
            super().__init__()
            self.results = results_list

        def on_epoch_begin(self, net, dataset_train, dataset_valid):
            k_value = net.module_.K  # Access the current K value
            print(f"Training for K = {k_value}")

        def on_epoch_end(self, net, dataset_train, dataset_valid):
            train_loss = net.history[-1, "train_loss"]
            valid_loss = net.history[-1, "valid_loss"]
            num_epochs = len(net.history)

            result = {
                "K": net.module_.K,
                "TrainLoss": train_loss,
                "ValidLoss": valid_loss,
                "NumEpochs": num_epochs,
                "FinalModelConfig": net.module_,
            }

            self.results.append(result)

    record_callback = RecordResultsCallback(results_list=[])

    model = MF_Bias(n_users=U_size, n_items=I_size, K=NUM_FEATURES, G_b=G_b)
    regressor = NeuralNetRegressor(
        module=model,
        module__n_users=U_size,
        module__n_items=I_size,
        optimizer=optim.Adam,
        optimizer__lr=LEARNING_RATE,
        criterion=nn.MSELoss,
        max_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[record_callback],
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
    )
    grid_search.fit(x, y)
    results_df = pd.DataFrame(record_callback.results)

    # Save this as well
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
