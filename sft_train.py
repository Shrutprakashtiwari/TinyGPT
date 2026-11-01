import torch
import torch.nn.functional as F
import tiktoken
import math
from model import GPT, vocab_size, embed_size, num_heads, block_size
from sft_data import input_ids, labels, num_batches, get_batch, attention_masks

device = torch.device("cpu")
enc = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def estimate():
    model.eval()
    out = {"train": 0.0, "val": 0.0}

    for split in ["train", "val"]:
        for k in range(100):
            x, y, mask = get_batch(split)
            logits, _, _ = model(x, attention_mask=mask)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                y[:, 1:].reshape(-1),
                ignore_index=-100
            )
            out[split] += loss.item()

        out[split] /= 100

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

input_ids = input_ids.to(device)
labels = labels.to(device)
attention_masks = attention_masks.to(device)

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.embed.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.trans.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.ln3.parameters(),
            "lr": 1e-4
        }
    ],
    weight_decay=0.01
)
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = param_group['lr']
start_step = 0
max_steps = 10000
warmup_steps = 500
max_lr = 3e-4
min_lr = 3e-5

model.train()

epochs = 10
best_val_loss = float('inf')
patience_counter = 0
patience = 3
accumulation_steps = 4

total_loss = 0.0
optimizer.zero_grad()


for step in range(start_step, max_steps):
    if step < warmup_steps:
        lr_multiplier = step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        decay_factor = 0.5 * (1 + math.cos(math.pi * progress))

        min_multiplier = min_lr / max_lr
        lr_multiplier = min_multiplier + (1 - min_multiplier) * decay_factor
    x, y, mask = get_batch("train")
    extended_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -10000.0

    logits, _, _ = model(x, attention_mask=mask)

    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        y[:, 1:].reshape(-1),
        ignore_index=-100,
        label_smoothing=0.1
    )
    total_loss += loss.item()

    loss = loss / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()


    if (step + 1) % 100 == 0:
        losses = estimate()

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            patience_counter = 0
            torch.save(model.state_dict(), "gpt_sft.pth")
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Val not improving")
                break

    print(
        # f"Epoch {epoch + 1}/{epochs}, "
        f"loss: {total_loss / (num_batches):.4f} ,"
        f"Val loss: {losses['val']:.4f}"
    )

model.load_state_dict(
    torch.load("gpt_sft.pth", map_location=device)
)

print("SFT model saved to gpt_sft.pth")
