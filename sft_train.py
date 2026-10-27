import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, vocab_size, embed_size, num_heads, block_size

device = torch.device("cpu")
enc = tiktoken.get_encoding("gpt2")

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

from sft_data import input_ids, labels

input_ids = input_ids.to(device)
labels = labels.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)

model.train()
epochs = 3

for epoch in range(epochs):
    total_loss = 0.0

    for i in range(len(input_ids)):
        x = input_ids[i:i + 1]
        y = labels[i:i + 1]

        logits, _, _ = model(x)
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

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"loss: {total_loss / len(input_ids):.4f}"
    )
torch.save(model.state_dict(), "gpt_sft.pth")
print("SFT model saved to gpt_sft.pth")
