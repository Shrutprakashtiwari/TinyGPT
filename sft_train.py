
import torch
import torch.nn.functional as F
import math

from model import GPT, vocab_size, embed_size, num_heads, block_size
from sft_data import get_batch


device = torch.device("cpu")

rank = 8
alpha = 16

checkpoint = torch.load("gpt_best.pth", map_location=device)
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=4
).to(device)


def load_converted_checkpoint(model, checkpoint, headsize):

    model_state = model.state_dict()
    loaded_keys = set()
    unexpected_keys = []

    for key, value in checkpoint.items():

        if key.endswith("qkv.weight"):

            prefix = key[:-len("qkv.weight")]

            if value.shape[0] != 3 * headsize:
                raise ValueError(
                    f"Unexpected QKV weight shape: {key}, {value.shape}"
                )

            q_weight, k_weight, v_weight = value.split(
                headsize, dim=0
            )

            new_weights = {
                prefix + "q.weight": q_weight,
                prefix + "k.weight": k_weight,
                prefix + "v.weight": v_weight
            }

            for new_key, tensor in new_weights.items():

                if new_key not in model_state:
                    raise KeyError(
                        f"New model does not contain {new_key}"
                    )

                if model_state[new_key].shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {new_key}: "
                        f"{model_state[new_key].shape} vs {tensor.shape}"
                    )

                model_state[new_key] = tensor
                loaded_keys.add(new_key)

        elif key.endswith("qkv.bias"):

            prefix = key[:-len("qkv.bias")]

            if value.shape[0] != 3 * headsize:
                raise ValueError(
                    f"Unexpected QKV bias shape: {key}, {value.shape}"
                )

            q_bias, k_bias, v_bias = value.split(
                headsize, dim=0
            )

            new_biases = {
                prefix + "q.bias": q_bias,
                prefix + "k.bias": k_bias,
                prefix + "v.bias": v_bias
            }

            for new_key, tensor in new_biases.items():

                if new_key not in model_state:
                    raise KeyError(
                        f"New model does not contain {new_key}"
                    )

                if model_state[new_key].shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {new_key}"
                    )

                model_state[new_key] = tensor
                loaded_keys.add(new_key)

        else:

            if key not in model_state:
                unexpected_keys.append(key)
                continue

            if model_state[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{model_state[key].shape} vs {value.shape}"
                )

            model_state[key] = value
            loaded_keys.add(key)
    missing_base_keys = [
        key for key in model_state
        if "lora" not in key.lower()
        and key not in loaded_keys
    ]

    if unexpected_keys or missing_base_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {unexpected_keys}\n"
            f"Missing pretrained parameters: {missing_base_keys}"
        )

    model.load_state_dict(model_state, strict=True)

    print("Pretrained checkpoint converted and loaded successfully!")


headsize = embed_size // num_heads

load_converted_checkpoint(model, checkpoint, headsize)


for param in model.parameters():
    param.requires_grad = False

for param in model.embed.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():

    if "lora" in name.lower():
        param.requires_grad = True


qkv_lora_params = []
o_lora_params = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    name = name.lower()

    if any(key in name for key in ["qlora", "klora", "vlora"]):
        qkv_lora_params.append(param)

    elif "olora" in name:
        o_lora_params.append(param)


assert qkv_lora_params, "No Q/K/V LoRA parameters found!"
assert o_lora_params, "No O LoRA parameters found!"


trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]

print("\nTrainable parameters:")

for name, param in model.named_parameters():

    if param.requires_grad:
        print(name, param.shape)

total_params = sum(p.numel() for p in model.parameters())
trainable_count = sum(p.numel() for p in trainable_params)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_count:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_count / total_params:.4f}%"
)


optimizer = torch.optim.AdamW(
    [
        {
            "params": qkv_lora_params,
            "lr": 3e-4
        },
        {
            "params": o_lora_params,
            "lr": 3e-4
        },
        {
            "params": model.embed.parameters(),
            "lr": 1e-5
        }
    ],
    weight_decay=0.01
)

for param_group in optimizer.param_groups:
    param_group["initial_lr"] = param_group["lr"]


start_step = 0
max_steps = 10000

warmup_steps = 500

max_lr = 3e-4
min_lr = 3e-5

accumulation_steps = 4
eval_interval = 100

best_val_loss = float("inf")

patience_counter = 0
patience = 3

total_loss = 0.0

optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def estimate():

    was_training = model.training
    model.eval()

    out = {
        "train": 0.0,
        "val": 0.0
    }

    for split in ["train", "val"]:

        for _ in range(100):

            x, y, mask = get_batch(split)

            logits, _, _ = model(
                x,
                attention_mask=mask
            )

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                y[:, 1:].reshape(-1),
                ignore_index=-100
            )

            out[split] += loss.item()

        out[split] /= 100

    model.train(was_training)

    return out



model.train()

for step in range(start_step, max_steps):

    # Learning-rate warmup
    if step < warmup_steps:

        lr_multiplier = step / warmup_steps

    # Cosine decay
    else:

        progress = (
            (step - warmup_steps) /
            (max_steps - warmup_steps)
        )

        decay_factor = 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        min_multiplier = min_lr / max_lr

        lr_multiplier = (
            min_multiplier +
            (1 - min_multiplier) * decay_factor
        )

    # Apply scheduler to every optimizer group
    for param_group in optimizer.param_groups:

        param_group["lr"] = (
            param_group["initial_lr"] * lr_multiplier
        )

    # Get training batch
    x, y, mask = get_batch("train")

    # Forward pass
    logits, _, _ = model(
        x,
        attention_mask=mask
    )

    # SFT loss
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        y[:, 1:].reshape(-1),
        ignore_index=-100,
        label_smoothing=0.1
    )

    total_loss += loss.item()
    (loss / accumulation_steps).backward()

    if (step + 1) % accumulation_steps == 0:

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            1.0
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if (step + 1) % eval_interval == 0:

        losses = estimate()

        avg_train_loss = total_loss / eval_interval
        total_loss = 0.0

        print(
            f"Step: {step + 1}, "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {losses['val']:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.7f}"
        )

        if losses["val"] < best_val_loss:

            best_val_loss = losses["val"]
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "gpt_sft.pth"
            )

            print("Best checkpoint saved!")

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print("Validation loss not improving. Early stopping.")
                break



model.load_state_dict(
    torch.load("gpt_sft.pth", map_location=device)
)

print("\nSFT training completed.")
print("Best validation loss:", best_val_loss)
print("Model saved to gpt_sft.pth")
