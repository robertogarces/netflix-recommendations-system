"""NeuMF-style model architecture for rating prediction.

Combines two complementary collaborative filtering signals in one forward pass:

  GMF (Generalized Matrix Factorization)
    dot product of user and item embeddings — equivalent to classic Matrix
    Factorization. It captures *linear* co-occurrence patterns:
    "user U tends to like items similar to what item I represents."

  MLP (Multi-Layer Perceptron)
    a small neural net applied to the *concatenation* of user and item embeddings.
    Concatenation keeps each dimension separate before mixing — this lets the
    network learn non-linear interactions that the dot product misses
    (e.g. "user U dislikes long films even when they like that genre").

Final prediction formula:
    ŷ = global_mean + user_bias + item_bias + dot(u, i) + MLP(u ⊕ i)

The bias terms shift predictions toward each user's/item's average rating,
so GMF+MLP only need to learn the *residual* above that baseline.

Reference: He et al. 2017, "Neural Collaborative Filtering" (WWW '17).
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
        # Embedding = lookup table of shape (num_entities, emb_size).
        # model.user_emb(user_idx) returns a vector of size emb_size that
        # encodes that user's learned "taste profile".
        self.user_emb  = nn.Embedding(num_users, emb_size)
        self.item_emb  = nn.Embedding(num_items, emb_size)

        # Scalar offset per user/item, separate from the embedding vectors.
        # user_bias captures "this user rates everything +0.3 above average".
        # item_bias captures "this film is universally liked / polarizing".
        # Embedding(..., 1) means each entry is a 1-D vector (a single scalar).
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # register_buffer: global_mean is stored in the checkpoint and moves
        # to the right device with model.to(device), but is NOT a trainable
        # parameter — it is computed once from training data and kept frozen.
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        # Build MLP layers dynamically from hidden_dims.
        # Input size = emb_size * 2 because we concatenate user and item vectors.
        # Each layer: Linear → ReLU activation → Dropout (regularization).
        # The final Linear projects from the last hidden size down to 1 scalar.
        layers, in_dim = [], emb_size * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output: one score per (user, item)
        self.mlp = nn.Sequential(*layers)

        # Small initialization keeps early predictions near global_mean.
        # If embeddings start large, the model produces garbage predictions and
        # wastes the first epochs recovering. std=0.01 means the initial dot
        # product (GMF) contributes ≈0 — the model starts as a pure bias model
        # and gradually learns richer signals as gradients flow in.
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user: LongTensor of user indices, shape (batch_size,)
            item: LongTensor of item indices, shape (batch_size,)

        Returns:
            Predicted ratings, shape (batch_size,)
        """
        u = self.user_emb(user)   # (batch_size, emb_size) — user taste vectors
        i = self.item_emb(item)   # (batch_size, emb_size) — item feature vectors

        # GMF: element-wise product, then sum across the embedding dimension.
        # If u and i point in the same direction, their dot product is large → high score.
        gmf = (u * i).sum(dim=1)  # (batch_size,)

        # MLP: concatenate [u, i] → shape (batch_size, emb_size*2), then pass through
        # the neural net. squeeze(-1) removes the trailing dim=1 that Linear(..., 1) adds.
        mlp = self.mlp(torch.cat([u, i], dim=1)).squeeze(-1)  # (batch_size,)

        # Combine all signals. squeeze(-1) on bias embeddings: Embedding(..., 1)
        # returns shape (batch, 1), so we flatten to (batch,) before adding.
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


def load_checkpoint(path, device: torch.device) -> tuple["NCF", dict]:
    """Rebuild the model from a self-contained checkpoint (see src.train).

    Returns the model in eval mode on `device` and the raw checkpoint dict
    (which also carries user2idx/item2idx/global_mean/mlflow_run_id).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["ncf_config"]
    model = NCF(
        num_users=len(ckpt["user2idx"]),
        num_items=len(ckpt["item2idx"]),
        emb_size=cfg["emb_size"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        global_mean=ckpt["global_mean"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
