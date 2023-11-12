from sklearn.decomposition import LatentDirichletAllocation

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor


class MF(nn.Module):
    def __init__(self, n_users, n_items, K, dropout=0):
        super().__init__()
        self.user_m = nn.Embedding(
            n_users, K, dtype=torch.float64
        )  # can include option sparse = True for memory
        self.item_m = nn.Embedding(n_items, K, dtype=torch.float64)
        self.drop_u = nn.Dropout(dropout)
        self.drop_i = nn.Dropout(dropout)

    def forward(self, x):
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        user_embeds = self.drop_u(self.user_m(user_ids))
        item_embeds = self.drop_i(self.item_m(item_ids))
        prod = user_embeds * item_embeds

        out = torch.sum(prod, 1)

        return out


# Matrix factorization with user/item biases
class MF_Bias(MF):
    def __init__(self, n_users, n_items, K, G_b, dropout=0):
        super().__init__(n_users, n_items, K, dropout)

        self.user_b = nn.Embedding(n_users, 1, dtype=torch.float64)
        self.item_b = nn.Embedding(n_items, 1, dtype=torch.float64)
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


class LDANet(NeuralNetRegressor):
    def __init__(self, n_topics, document_word, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_word = document_word
        self.LDA = LatentDirichletAllocation(n_topics).fit(document_word)
        print("Done fitting")

    def get_loss(self, y_pred, y_true, X=None, training=False):
        print("Calculating loss")
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss = loss + self.LDA._perplexity_precomp_distr(
            self.document_word,
            doc_topic_distr=self.module.get_item_embeds().clone().detach().numpy(),
        )
        print(loss)
        return loss
