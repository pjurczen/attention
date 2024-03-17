import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
vocab_size = 65
batch_size = 32
n_embed = 64
block_size = 8
num_iters = 10000
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3

with open('../data/tinyshakespeare/input.txt', 'r', encoding='UTF-8') as f:
    text = f.read()
dictionary = sorted(list(set(text)))
vocab_size = len(dictionary)
stoi = {s:i for i,s in enumerate(dictionary)}
itos = {i:s for i,s in enumerate(dictionary)}
encode = lambda s: [stoi[x] for x in s]
decode = lambda l: ''.join(itos[x] for x in l)
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.int64)

n = int(0.9*len(data))
train = data[:n]
test = data[n:]


def get_batch(split):
    data = train if split == 'train' else test
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)  # (C, n_embed)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embed)  # (C, n_embed)

    def forward(self, x, y=None):  # x: (B, T), y: (B, T)
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        logits = self.token_embedding_table(x)  # (B, T, C)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(x)
            logits = logits[:, -1, :]  # (B, 1, C)
            probs = F.softmax(logits, dim=-1)
            yhat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, yhat), dim=1)
        return x


model = BigramLanguageModel(vocab_size).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(num_iters):
    if i % eval_iters == 0 or i == num_iters - 1:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(model.generate(torch.zeros((1,1), dtype=torch.int64, device=device), max_tokens=100)[0].tolist()))
