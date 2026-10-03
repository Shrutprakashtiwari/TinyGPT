import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
import tiktoken
import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

data = torch.tensor(encode(text), dtype=torch.long)

batch_size = 32
block_size = 64
embed_size = 128
num_heads = 4
head_size = embed_size // num_heads

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class SelfAttention(nn.Module):
    def __init__(self, embed_size, headsize):
        super().__init__()
        self.qkv = nn.Linear(embed_size, 3 * headsize)
        self.headsize = headsize
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.headsize, dim=-1)
    
        w = q @ k.transpose(-2, -1)
        w = w / math.sqrt(self.headsize)

        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        mask = ~mask

        w = w.masked_fill(mask, float('-inf'))
        w = w.softmax(dim=-1)
        w = self.dropout(w)

        out = w @ v
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, headsize, embed_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_size, headsize)
            for _ in range(num_heads)
        ])
        self.proj=nn.Linear(num_heads*headsize, embed_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        res=torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(res))
class FeedForward(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.l1=nn.Linear(embed_size, 4*embed_size)
        self.gelu=nn.GELU()
        self.l2=nn.Linear(4*embed_size, embed_size)
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        x=self.l1(x)
        x=self.gelu(x)
        x=self.l2(x)
        x=self.dropout(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.ma = MultiHeadAttention(embed_size // num_heads, embed_size, num_heads)
        self.ln1=nn.LayerNorm(embed_size)
        self.ff=FeedForward(embed_size)
        self.ln2=nn.LayerNorm(embed_size)
    def forward(self, x):
        x=x+self.ma(self.ln1(x))
        x=x+self.ff(self.ln2(x))
        return x
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, block_size, num_layers):
        super().__init__()

        self.position = nn.Embedding(block_size, embed_size)

        self.trans = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads)
            for _ in range(num_layers)
        ])

        self.ln3 = nn.LayerNorm(embed_size)
        self.logits.weight = self.embed.weight
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape

        tok = self.embed(x)
        pos = self.position(torch.arange(T, device=x.device)).unsqueeze(0)

        x = tok + pos
        x = self.trans(x)
        x = self.ln3(x)
        x = self.logits(x)

        return x
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            with torch.no_grad():
                logits = self(idx_cond)
            
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            
            # probs = F.softmax(logits / temperature, dim=-1)

            probs = F.softmax(logits / temperature, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative_probs > 0.9
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(sorted_probs, num_samples=1)
            idx_next = torch.gather(sorted_indices, -1, idx_next)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx



model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=4
    )

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4
)

for step in range(10000):

    x, y = get_batch()

    logits = model(x)

    loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    y.view(-1)
)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())
start = "Hello"
idx = torch.tensor([encode(start)], dtype=torch.long)

out = model.generate(idx, 100,temperature=0.7)
print(decode(out[0].tolist()))
torch.save(model.state_dict(), "gpt.pth")
model.load_state_dict(torch.load("gpt.pth"))
