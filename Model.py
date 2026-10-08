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
    def __init__(self, embed_size, headsize,block_size):
        super().__init__()
        self.qkv = nn.Linear(embed_size, 3 * headsize)
        self.headsize = headsize
        self.block_size = block_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, k_cache=None, v_cache=None):
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k_new, v_new = qkv.split(self.headsize, dim=-1)

        if k_cache is not None and v_cache is not None:
            k=torch.cat([k_cache, k_new], dim=1)
            v=torch.cat([v_cache, v_new], dim=1)
            if k.size(1)>self.block_size:
                k=k[:,-self.block_size:,:]
                v=v[:,-self.block_size:,:]
        else:
            k = k_new
            v = v_new
        T_k=k.size(1)
        if k_cache is None:
            q_start = 0
        else:
            q_start = k_cache.size(1)
        freq = 1 / (10000 ** (torch.arange(0, self.headsize, 2, device=x.device) / self.headsize))
        if k_cache is None:
            q_start=0
        else:
            q_start=k_cache.size(1)
        q_positions=torch.arange(q_start, q_start + T, device=x.device)
        k_positions=torch.arange(0,T_k, device=x.device)
        # print("q_positions:", q_positions)
        # print("k_positions:", k_positions)
        q_theeta=q_positions[:, None] * freq[None, :]
        k_theeta=k_positions[:, None] * freq[None, :]
        q_cos=torch.cos(q_theeta)
        q_sin=torch.sin(q_theeta)
        k_cos=torch.cos(k_theeta)
        k_sin=torch.sin(k_theeta)
        q_even = q[:, :, ::2]
        q_odd  = q[:, :, 1::2]

        q_rot_even = q_even * q_cos - q_odd * q_sin
        q_rot_odd  = q_even * q_sin + q_odd * q_cos
        k_even = k[:, :, ::2]
        k_odd  = k[:, :, 1::2]
        k_rot_even = k_even * k_cos - k_odd * k_sin
        k_rot_odd  = k_even * k_sin + k_odd * k_cos
        q_rot=torch.stack([q_rot_even, q_rot_odd], dim=-1).reshape(B, T, self.headsize)
        k_rot=torch.stack([k_rot_even, k_rot_odd], dim=-1).reshape(B, T_k, self.headsize)
        w = q_rot @ k_rot.transpose(-2, -1)
        w = w / math.sqrt(self.headsize)
        if k_cache is None:
            mask = torch.tril(torch.ones(T, T_k, device=x.device)).bool()
            mask = ~mask

            w = w.masked_fill(mask, float('-inf'))
        w = w.softmax(dim=-1)
        w = self.dropout(w)

        out = w @ v
        return out,k,v
class MultiHeadAttention(nn.Module):
    def __init__(self, headsize, embed_size, num_heads, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_size, headsize, block_size)
            for _ in range(num_heads)
        ])
        self.proj=nn.Linear(num_heads*headsize, embed_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, k_cache=None, v_cache=None):
        k_cache_new=[]
        v_cache_new=[]
        out_heads=[]
        for i,head in enumerate(self.heads):
            if k_cache is not None and v_cache is not None:
                k_c=k_cache[i]
                v_c=v_cache[i]
            else:
                k_c=None
                v_c=None
            out,k,v=head(x,k_c,v_c)
            out_heads.append(out)
            k_cache_new.append(k)
            v_cache_new.append(v)
        res=torch.cat(out_heads, dim=-1)
        return self.dropout(self.proj(res)),k_cache_new,v_cache_new
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
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.ma = MultiHeadAttention(embed_size // num_heads, embed_size, num_heads, block_size)
        self.ln1=nn.LayerNorm(embed_size)
        self.ff=FeedForward(embed_size)
        self.ln2=nn.LayerNorm(embed_size)
    def forward(self, x, k_cache=None, v_cache=None):
        attn_out, k, v = self.ma(self.ln1(x), k_cache, v_cache)
        x=x+attn_out
        x=x+self.ff(self.ln2(x))
        return x, k, v
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, block_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)


        self.trans = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, block_size)
            for _ in range(num_layers)
        ])

        self.ln3 = nn.LayerNorm(embed_size)
        self.logits = nn.Linear(embed_size, vocab_size)
        print(self.embed.weight.std().item())
        self.logits.weight = self.embed.weight
        self.block_size = block_size

    def forward(self, x,k_cache=None,v_cache=None):
        new_k_cache=[]
        new_v_cache=[]

        B, T = x.shape

        x=self.embed(x)
        for i, block in enumerate(self.trans):
            k_c = k_cache[i] if k_cache is not None else None
            v_c = v_cache[i] if v_cache is not None else None
            x, k, v = block(x, k_c, v_c)
            new_k_cache.append(k)
            new_v_cache.append(v)
        x = self.ln3(x)
        # x = self.logits(x)
        logits=self.logits(x)

        return logits, new_k_cache, new_v_cache
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()

        k_cache = None
        v_cache = None

        for _ in range(max_new_tokens):

            if k_cache is None:
                idx_cond = idx
            else:
                if k_cache[0][0].size(1) >= self.block_size:
                    k_cache=None
                    v_cache=None
                    idx_cond = idx[:, -self.block_size:]
                else:
                    idx_cond = idx[:, -1:]
            with torch.no_grad():
                logits, k_cache, v_cache = self(idx_cond, k_cache, v_cache)

            logits = logits[:, -1, :]
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
x = torch.randint(0, vocab_size, (2, 8))

# logits, k_cache, v_cache = model(x)

# print("logits:", logits.shape)
# print("K:", k_cache[0][0].shape)
# print("V:", v_cache[0][0].shape)
# x_next = torch.randint(0, vocab_size, (2, 1))

# logits2, k_cache2, v_cache2 = model(
#     x_next,
#     k_cache,
#     v_cache
# )

# print("cached logits:", logits2.shape)
# print("cached K:", k_cache2[0][0].shape)
# print("cached V:", v_cache2[0][0].shape)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4
)

for step in range(10000):

    x, y = get_batch()

    logits,_,_ = model(x)
    # print("logits:", logits.min().item(), logits.max().item(), logits.mean().item())

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
