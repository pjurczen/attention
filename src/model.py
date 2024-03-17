import torch
import torch.nn as nn
from torch.nn import functional as F
import math
torch.manual_seed(1337)

vocab_size = 65
block_size = 32
n_embed = 256
head_size = 32
n_head = 4
n_blocks = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1


class SingleAttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # T, T

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        
        wei = k @ q.transpose(-2, -1) * head_size**-0.5  # B, T, T
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -math.inf)
        wei = F.softmax(wei, dim=-1)  # softmax over channels (last dimension)
        v = self.value(x)  # B, T, head_size
        
        out = wei @ v  # B, T, head_size
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleAttentionHead(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.layer_norm_sa = nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm_ff = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.layer_norm_sa(x + self.self_attention(x))
        x = self.layer_norm_ff(x + self.feed_forward(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)  # (C, n_embed)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embed)  # (C, n_embed)
        self.transformers = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embed, vocab_size)

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
            x_cond = x[:, -block_size:]
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]  # (B, 1, C)
            probs = F.softmax(logits, dim=-1)
            yhat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, yhat), dim=1)
        return x

