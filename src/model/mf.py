import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, K):
        super().__init__()
        self.user_m = nn.Embedding(n_users, K, dtype = torch.float64) # can include option sparse = True for memory
        self.item_m = nn.Embedding(n_items, K, dtype = torch.float64)
    
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