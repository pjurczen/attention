import torch
from model import BigramLanguageModel, device, block_size

num_iters = 5000
eval_iters = 200
learning_rate = 3e-4
batch_size = 16

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



model = BigramLanguageModel().to(device)
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

print(decode(model.generate(torch.zeros((1,1), dtype=torch.int64, device=device), max_tokens=1000)[0].tolist()))
