import numpy as np
import pandas as pd
import pickle

import torch
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skorch.callbacks import EpochScoring

from model.mf import MF_Bias, LDANet
from model.utility import RMSELoss, RMSE
from train.saving import save_model_results

BATCH_SIZE = 8192
NUM_FEATURES = 5
LEARNING_RATE = 0.01
EPOCHS = 15
DECAY = 1e-3
DROPOUT = 0.4
LDA_ALPHA = 0.001
# LDA_ALPHA = 0

if __name__ == '__main__':

    # KEYWORDS:
    # games
    # games_apps
    # games_health
    # all
    keyword = 'all'

    # Loads the model
    model_path = f'../results/model_{keyword}_bs8192K5LR0.01E15D0.001DR0.4ALPH0.001.pkl'
    print(model_path)
    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)

    # TF-IDF
    review_data_df = pd.read_csv(f'../datasets/processed/{keyword}_I_reviews.csv', escapechar = '\\')
    review_data = review_data_df['reviewText'].tolist()
    vectorizer = TfidfVectorizer(encoding="utf-8", lowercase=True)
    document_word = vectorizer.fit_transform(review_data)

    # Loads the user an item lists - corresponding to the latent factors in MF
    # Note: these are IDs AND NOT THE ACTUAL NAMES - you will have to grab these from the raw datasets
    UILists = np.load(f'../datasets/processed/{keyword}_UI_lists.npz', allow_pickle = True)
    U = UILists['u']
    I = UILists['i']

    # Gets embeddings from the model
    user_embeds = model.module.get_user_embeds()
    item_embeds = model.module.get_item_embeds()

    # data = np.load(f'../datasets/processed/{keyword}_ratings.npz')
    # review_data_df = pd.read_csv(f'../datasets/processed/{keyword}_I_reviews.csv', escapechar = '\\')
    # print(len(review_data_df))
    # review_data = review_data_df['reviewText'].tolist()
    # x = data['x']
    # y = data['y']
    # U_size = data['U_size']
    # I_size = data['I_size']
    # G_b = data['G_b']
    
    # print(len(x))
    # print(len(y))
    # print(U_size)
    # print(I_size)
    # print(G_b)

    # # LDA
    # vectorizer = TfidfVectorizer(encoding="utf-8", lowercase=True)
    # document_word = vectorizer.fit_transform(review_data)
    # # document_word = None

    # # Model/training functions
    # model = MF_Bias(U_size, I_size, NUM_FEATURES, G_b, DROPOUT)
    # loss_fn = RMSELoss()
    # optimizer = Adam

    # regressor = LDANet(
    #     NUM_FEATURES,
    #     LDA_ALPHA,
    #     document_word,
    #     model,
    #     criterion = loss_fn,
    #     optimizer = optimizer,
    #     optimizer__param_groups = [
    #         ('user_m.weight', {'weight_decay': DECAY}),
    #         ('item_m.weight', {'weight_decay': DECAY})
    #     ],
    #     optimizer__lr = LEARNING_RATE,
    #     batch_size = BATCH_SIZE,
    #     max_epochs = EPOCHS,
    #     callbacks = [
    #         ('train_rmse', EpochScoring(RMSE, name = 'train_rmse', on_train = True)),
    #         ('test_rmse', EpochScoring(RMSE, name = 'test_rmse'))
    #     ]
    # )
    
    # save_model_results(regressor, x, y, f'{keyword}_bs{BATCH_SIZE}K{NUM_FEATURES}LR{LEARNING_RATE}E{EPOCHS}D{DECAY}DR{DROPOUT}ALPH{LDA_ALPHA}')