
import torch
import tiktoken

from model import GPT, vocab_size, embed_size, num_heads, block_size

device = torch.device("cpu")
enc = tiktoken.get_encoding("gpt2")

# Recreate the same model architecture
model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=4
).to(device)

# Load your fine-tuned weights
model.load_state_dict(
    torch.load("gpt_sft.pth", map_location=device)
)
model.eval()

print("\nTinyGPT is ready! Type 'exit' to quit.\n")

while True:
    user_message = input("You: ").strip()

    if user_message.lower() in ("exit", "quit"):
        print("TinyGPT: Goodbye!")
        break

    if not user_message:
        continue

    # Match the format used during SFT
    prompt = f"user:{user_message}\nassistant:"
    prompt_tokens = enc.encode(prompt)

    # Leave room for generated tokens
    prompt_tokens = prompt_tokens[-(block_size - 1):]
    input_ids = torch.tensor(
        [prompt_tokens], dtype=torch.long, device=device
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.6
        )

    # Decode only the newly generated tokens
    new_tokens = output[0, len(prompt_tokens):].tolist()

    # Stop displaying at the end-of-text token, if generated
    if enc.eot_token in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(enc.eot_token)]

    reply = enc.decode(new_tokens).strip()
    print(f"TinyGPT: {reply}\n")
