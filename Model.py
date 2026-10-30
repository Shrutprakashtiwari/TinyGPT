import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
import tiktoken
import requests
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

# use_bf16 = (
#     device.type == "cuda"
#     and torch.cuda.is_bf16_supported()
# )

# print("Device:", device)
# print("Using BF16:", use_bf16)

# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# text = requests.get(url).text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

batch_size = 32
block_size = 128
embed_size = 128
num_heads = 4
head_size = embed_size // num_heads

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model,get_batch, vocab_size):
    model.eval()
    out={"train":0.0, "val":0.0}
    for split in ["train", "val"]:
        for k in range(100):
            x,y=get_batch(split)
            logits,_,_ = model(x)
            loss=F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            out[split]+=loss.item()
        out[split]/=100
    model.train()
    return out
class RMSNorm(nn.Module):
    def __init__(self,embed_size, eps=1e-8):
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(embed_size))
    def forward(self,x):
        mean_sq=x.pow(2).mean(dim=-1, keepdim=True)
        rms=torch.sqrt(mean_sq+self.eps)
        x_norm=x/rms
        return self.scale * x_norm
class SelfAttention(nn.Module):
    def __init__(self, embed_size, headsize,block_size):
        super().__init__()
        self.qkv = nn.Linear(embed_size, 3 * headsize)
        self.headsize = headsize
        self.block_size = block_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, k_cache=None, v_cache=None ,attention_mask=None):
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
            causal_mask = torch.tril(torch.ones(T, T_k, device=x.device)).bool()
            if attention_mask is not None:
                padding_mask = attention_mask[:, None, None, :].bool()
                combined_mask = causal_mask[None, None, :, :] & padding_mask
            else:
                combined_mask = causal_mask
            # combined_mask=causal_mask& padding_mask

            w = w.masked_fill(~combined_mask, float('-inf'))
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
    def forward(self, x, k_cache=None, v_cache=None, attention_mask=None):
        k_cache_new=[]
        v_cache_new=[]
        out_heads=[]
        # mask=attention_mask
        for i,head in enumerate(self.heads):
            if k_cache is not None and v_cache is not None:
                k_c=k_cache[i]
                v_c=v_cache[i]
            else:
                k_c=None
                v_c=None
            out,k,v=head(x,k_c,v_c, attention_mask=attention_mask)
            out_heads.append(out)
            k_cache_new.append(k)
            v_cache_new.append(v)
        res=torch.cat(out_heads, dim=-1)
        return self.dropout(self.proj(res)),k_cache_new,v_cache_new
class FeedForward(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.gate=nn.Linear(embed_size, 4*embed_size)
        self.up=nn.Linear(embed_size, 4*embed_size)
        self.down=nn.Linear(4*embed_size, embed_size)
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        gate=self.gate(x)
        up=self.up(x)
        x=F.silu(gate)*up
        x=self.down(x)
        x=self.dropout(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.ma = MultiHeadAttention(embed_size // num_heads, embed_size, num_heads, block_size)
        self.ln1=RMSNorm(embed_size)
        self.ff=FeedForward(embed_size)
        self.ln2=RMSNorm(embed_size)
    def forward(self, x, k_cache=None, v_cache=None, attention_mask=None):
        attn_out, k, v = self.ma(self.ln1(x), k_cache, v_cache, attention_mask=attention_mask)
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

        self.ln3 = RMSNorm(embed_size)
        self.logits = nn.Linear(embed_size, vocab_size)
        # print(self.embed.weight.std().item())
        self.logits.weight = self.embed.weight
        self.block_size = block_size

    def forward(self, x,k_cache=None,v_cache=None, attention_mask=None):
        new_k_cache=[]
        new_v_cache=[]

        B, T = x.shape

        x=self.embed(x)
        for i, block in enumerate(self.trans):
            k_c = k_cache[i] if k_cache is not None else None
            v_c = v_cache[i] if v_cache is not None else None
            x, k, v = block(x, k_c, v_c,attention_mask=attention_mask)
            new_k_cache.append(k)
            new_v_cache.append(v)
        x = self.ln3(x)
        # x = self.logits(x)
        logits=self.logits(x)

        return logits, new_k_cache, new_v_cache
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        eos_token_id=None,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    ):
        self.eval()

        k_cache = None
        v_cache = None

        for _ in range(max_new_tokens):

            if k_cache is None:
                idx_cond = idx[:, -self.block_size:]
            else:
                if k_cache[0][0].size(1) >= self.block_size:
                    k_cache = None
                    v_cache = None
                    idx_cond = idx[:, -self.block_size:]
                else:
                    idx_cond = idx[:, -1:]

            with torch.no_grad():
                logits, k_cache, v_cache = self(
                    idx_cond, k_cache, v_cache
                )

            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    previous_tokens = idx[b].unique()
                    previous_logits = logits[b, previous_tokens]

                    logits[b, previous_tokens] = torch.where(
                        previous_logits < 0,
                        previous_logits * repetition_penalty,
                        previous_logits / repetition_penalty
                    )

            # Greedy decoding
            if not do_sample:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                # Temperature
                logits = logits / max(temperature, 1e-5)

                # Top-k filtering
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))

                # Top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )

                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        sorted_mask, float('-inf')
                    )

                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None:
                if idx_next.shape[0] == 1 and idx_next.item() == eos_token_id:
                    break

        return idx


model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=4
    )
# x = torch.randint(0, vocab_size, (2, 8))

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
# x = torch.randn(2, 8, embed_size)
# out = FeedForward(embed_size)(x)

# print(out.shape)
if __name__ == "__main__":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    if os.path.exists("gpt_checkpoint.pth"):
        checkpoint = torch.load(
            "gpt_checkpoint.pth",
            map_location="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        checkpoint = torch.load("gpt_checkpoint.pth", map_location=device)
        start_step = checkpoint["step"] + 1

        print("Training checkpoint loaded successfully!")
        print("Saved step:", checkpoint["step"])
    else:
        checkpoint = None
        best_val_loss = float("inf")
        patience_counter = 0
        start_step = 0

    warmup_steps = 500
    max_lr = 3e-4
    min_lr = 3e-5
    max_steps = 10000
    patience = 4
    for step in range(start_step,  max_steps):
        if step<warmup_steps:
            lr=max_lr*step/warmup_steps
        else:
            progress=(step-warmup_steps)/(max_steps-warmup_steps)
            decay_factor=0.5*(1+math.cos(math.pi*progress))
            lr=min_lr+(max_lr-min_lr)*decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        x, y = get_batch(split='train')
    #     with torch.autocast(
    #     device_type="cpu",
    #     dtype=torch.bfloat16,
    #     enabled=True
    # ):

        logits,_,_ = model(x)
            # print("logits:", logits.min().item(), logits.max().item(), logits.mean().item())

        loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            losses = estimate_loss()

            print(
                step,
                "train:", losses["train"],
                "val:", losses["val"]
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), "gpt_best.pth")
                print("Best model saved with val loss:", best_val_loss)
                patience_counter = 0
            else:
                patience_counter+=1
            
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }, "gpt_checkpoint.pth")
            import os

            print("Best model exists:", os.path.exists("gpt_best.pth"))
            print("Training checkpoint exists:", os.path.exists("gpt_checkpoint.pth"))
            if patience_counter>=patience:
                print("Early stopping triggered. No improvement in validation loss for", patience, "evaluations.")
                break


    start = "Hello"
    idx = torch.tensor([encode(start)], dtype=torch.long)
    model.load_state_dict(
        torch.load("gpt_best.pth", map_location="cpu")
    )

    model.eval()
    output = model.generate(
        idx,
        max_new_tokens=150,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=enc.eot_token
)
    print(decode(output[0].tolist()))
