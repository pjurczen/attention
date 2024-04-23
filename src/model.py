import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_blocks: int = 12
    n_head: int = 12
    n_embed: int = 384
    dropout: float = 0.1


class SingleAttentionHead(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head_size = config.n_embed // config.n_head
        self.key = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # T, T

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        
        wei = k @ q.transpose(-2, -1) * self.head_size**-0.5  # B, T, T
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -math.inf)
        wei = F.softmax(wei, dim=-1)  # softmax over channels (last dimension)
        v = self.value(x)  # B, T, head_size
        
        out = wei @ v  # B, T, head_size
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([SingleAttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.ReLU(),
            nn.Linear(4*config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm_sa = nn.LayerNorm(config.n_embed)
        self.feed_forward = FeedForward(config)
        self.layer_norm_ff = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = self.layer_norm_sa(x + self.self_attention(x))
        x = self.layer_norm_ff(x + self.feed_forward(x))
        return x


class GPTModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)  # (C, n_embed)
        self.position_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)  # (C, n_embed)
        self.transformers = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_blocks)])
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x, y=None):  # x: (B, T), y: (B, T)
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(x)  # (B, T, n_embed)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformers(x)
        logits = self.lm_head(x)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            x_cond = x[:, -self.config.block_size:]
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]  # (B, 1, C)
            probs = F.softmax(logits, dim=-1)
            yhat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, yhat), dim=1)
        return x

