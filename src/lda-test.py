import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from model.mf import MF_Bias, LDANet
from model.utility import RMSELoss
from train.saving import save_model_results

BATCH_SIZE = 64
NUM_FEATURES = 10
LEARNING_RATE = 0.0005
EPOCHS = 15
DECAY = 1e-3
DROPOUT = 0.4

if __name__ == '__main__':
    
    data = np.load('../datasets/processed/games_ratings.npz')
    review_data_df = pd.read_csv('../datasets/processed/games_I_reviews.csv', escapechar = '\\')
    print(len(review_data_df))
    review_data = review_data_df['reviewText'].tolist()
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

    # LDA
    vectorizer = TfidfVectorizer(encoding="utf-8", lowercase=True)
    document_word = vectorizer.fit_transform(review_data)

    # Train/test splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    y_train, y_test = torch.tensor(y_train, dtype = torch.float64), torch.tensor(y_test, dtype = torch.float64)

    # Model/training functions
    model = MF_Bias(U_size, I_size, NUM_FEATURES, G_b, DROPOUT)
    loss_fn = RMSELoss()
    optimizer = Adam

    regressor = LDANet(
        NUM_FEATURES,
        document_word,
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
    
    save_model_results(regressor, x, y, f'bs{BATCH_SIZE}K{NUM_FEATURES}LR{LEARNING_RATE}E{EPOCHS}D{DECAY}DR{DROPOUT}')