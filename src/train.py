import torch
from torch._prims_common import check_pin_memory
from model import BigramLanguageModel, ModelConfig
import os

out_dir = '../output'
init_from = 'scratch' # 'scratch' or 'resume'
num_iters: int = 5000
eval_iters: int = 200
learning_rate: float = 3e-4
batch_size: int = 32
block_size: int = 256
vocab_size: int = 65
n_blocks: int = 12
n_head: int = 12
n_embed: int = 192
dropout: float = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

os.makedirs(out_dir, exist_ok=True)

best_val_loss = 1e9
iter = 0

if init_from == 'scratch':
    print("Initialize new model")
    model_args = dict(block_size=block_size, vocab_size=vocab_size, n_blocks=n_blocks, n_head=n_head, n_embed=n_embed, dropout=dropout)
    model_config = ModelConfig(**model_args)
    model = BigramLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif init_from == 'resume':
    print("Resume training existing model")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_config = ModelConfig(**model_args)
    model = BigramLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter = checkpoint['iter']
    best_val_loss = checkpoint['best_val_loss']
    checkpoint = None
else:
    raise Exception(f"Mode {init_from} not supported")


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
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

while True:
    if iter % eval_iters == 0 or iter == num_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print("saving checkpoint")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    iter += 1

    if iter > num_iters:
        break

print(loss.item())

print(decode(model.generate(torch.zeros((1,1), dtype=torch.int64, device=device), max_tokens=1000)[0].tolist()))
