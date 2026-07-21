model.eval()

while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    idx = torch.tensor([encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=50)

    print("GPT:", decode(out[0].tolist()))
