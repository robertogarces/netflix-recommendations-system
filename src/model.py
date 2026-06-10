"""NeuMF-style model architecture for rating prediction.

Combines the two classic collaborative filtering signals:
- GMF path: dot product of user/item embeddings (the proven MF signal)
- MLP path: nonlinear interaction over concatenated embeddings

prediction = global_mean + user_bias + item_bias + dot(u, i) + MLP(u ⊕ i)
"""

import torch
from torch import nn


class NCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_size: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        global_mean: float = 0.0,
    ):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_size)
        self.item_emb  = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        layers, in_dim = [], emb_size * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Small init keeps early predictions near the global mean, so the model
        # starts from the best constant predictor and learns residuals.
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user)
        i = self.item_emb(item)
        gmf = (u * i).sum(dim=1)
        mlp = self.mlp(torch.cat([u, i], dim=1)).squeeze(-1)
        return (
            self.global_mean
            + self.user_bias(user).squeeze(-1)
            + self.item_bias(item).squeeze(-1)
            + gmf
            + mlp
        )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
