import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(
                config["context_length"],
                config["context_length"],
                dtype=torch.bool,
            )),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T], float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["embedding_dim"]
        self.network = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["embedding_dim"]
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["embedding_dim"]
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], dim)
        self.position_embedding = nn.Embedding(
            config["context_length"], dim
        )
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config["num_layers"])
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, config["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        if T > self.config["context_length"]:
            raise ValueError("Sequence exceeds context length.")
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)
