import torch
import torch.nn as nn
import torch.nn.functional as F
import math

with open("1661-0.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars=list(set(text.split()))
# print(char)
# chars=enumerate(char)
length=len(chars)
print(length)
stoi={word : i for i,word in enumerate(chars)}
# print(stoi)
itos={i : word for word,i in stoi.items()}
# print(itos)
# data = torch.tensor(encode(text), dtype=torch.long)
encode = lambda s: [stoi[word] for word in s.split()]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
batch_size = 32
block_size = 64
embed_dim = 64
num_heads = 4
head_size = embed_dim // num_heads
def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key= nn.Linear(embed_dim,head_size,bias=False)
        self.query=nn.Linear(embed_dim,head_size,bias=False)
        self.value=nn.Linear(embed_dim,head_size,bias=False)
    def forward(self,x):
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        w=q@k.transpose(-2,-1)
        d=math.sqrt(head_size)
        w=w/d
        y=torch.ones(block_size,block_size)
        y=torch.tril(y,diagonal=0)
        y=y.bool()
        y=~y
        w.masked_fill_(y,float('-inf'))
        w=w.softmax(dim=-1)
        score=w@v
        return score


class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(
            num_heads * head_size,
            embed_dim
        )
    def forward(self,x):
        out=torch.cat(
            [h(x) for h in self.heads],dim=-1
        )
        return self.proj(out)



class FeedForward(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.l1=nn.Linear(embed_dim,4*embed_dim)
        self.gelu=nn.GELU()
        self.l2=nn.Linear(4*embed_dim,embed_dim)

    def forward(self,x):
        x=self.l1(x)
        x=self.gelu(x)
        x=self.l2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.ma=MultiHeadAttention(num_heads,head_size)
        self.ln1=nn.LayerNorm(embed_dim)
        self.ff=FeedForward(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)

    def forward(self,x):

        x=x+self.ma(self.ln1(x))
        # x=self.ln1(x)

        x=x+self.ff(self.ln2(x))
        # x=self.ln2(x)

        return x


class GPT(nn.Module):
    def __init__(self,length,
                  embed_dim,
                  block_size,
                  num_heads,
                  

                  ):
        super().__init__()
        self.emb=nn.Embedding(length,embed_dim)
        self.position=nn.Embedding(block_size,embed_dim)
        self.trans=nn.Sequential(
            TransformerBlock(embed_dim, num_heads),
            TransformerBlock(embed_dim, num_heads)
        )
        self.ln3=nn.LayerNorm(embed_dim)
        self.logit=nn.Linear(embed_dim,length)
        self.block_size=block_size
    def forward(self,x):
        x=self.emb(x)
        pos=torch.arange(block_size, device=x.device)
        pos=self.position(pos)
        x=x+pos
        x=self.trans(x)
        x=self.ln3(x)
        x=self.logit(x)
        return x
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            return idx




model = GPT(
    length=length,
    embed_dim=embed_dim,
    block_size=block_size,
    num_heads=num_heads,
    )

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4
)

for step in range(1000):

    x, y = get_batch()

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, length),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

