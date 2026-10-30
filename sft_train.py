import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, vocab_size, embed_size, num_heads, block_size

device = torch.device("cpu")
enc = tiktoken.get_encoding("gpt2")
@torch.no_grad()
def estimate():
    model.eval()
    out={"train":0.0, "val":0.0}
    for split in ["train","val"]:
        for k in range(100):
            x,y,mask=get_batch(split)
            logits,_,_=model(x, attention_mask=mask)
            loss=F.cross_entropy(
                logits[:,:-1,:].reshape(-1,vocab_size),
                y[:,1:].reshape(-1),
                ignore_index=-100)
            out[split]+=loss.item()
        out[split]/=100
    model.train()
    return out
        

model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=4
).to(device)

model.load_state_dict(
    torch.load("gpt_best.pth", map_location=device)
)

print("Pretrained model loaded!")

from sft_data import input_ids, labels, num_batches,get_batch,attention_masks

input_ids = input_ids.to(device)
labels = labels.to(device)
attention_masks=attention_masks.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

model.train()
epochs = 10
best_val_loss=float('inf')
patience_counter=0
patience=3
accumulation_steps=4
for epoch in range(epochs):
    total_loss = 0.0

    for i in range((num_batches)):
        x,y,mask=get_batch("train")

        logits, _, _ = model(x,attention_mask=mask)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab_size),
            y[:, 1:].reshape(-1),
            ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    losses=estimate()
    if losses["val"]<best_val_loss:
        best_val_loss=losses["val"]
        patience_counter=0
        torch.save(model.state_dict(), "gpt_sft.pth")
    else:
        patience_counter+=1
    if patience_counter==patience:
        print("Val not improving")

        break
    

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"loss: {total_loss / (num_batches):.4f} ,"
        f"Val loss: {losses['val']:.4f}"
    )
model.load_state_dict(
    torch.load("gpt_sft.pth", map_location=device)
)
print("SFT model saved to gpt_sft.pth")
