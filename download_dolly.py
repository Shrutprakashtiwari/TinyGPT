
from datasets import load_dataset
import json

print("Downloading Dolly 15K...")

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

conversations = []

for item in dataset:
    instruction = item["instruction"].strip()
    context = item["context"].strip()
    response = item["response"].strip()

    if not instruction or not response:
        continue

    # Keep examples short enough for our initial TinyGPT experiment
    if len(instruction.split()) > 20 or len(response.split()) > 25:
        continue

    user_message = instruction

    if context:
        user_message += "\nContext: " + context

    conversations.append({
        "user": user_message,
        "assistant": response
    })

with open("dolly_conversations.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print(f"Saved {len(conversations)} short conversations.")
print("File: dolly_conversations.json")
