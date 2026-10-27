import json
import tiktoken
import torch
with open("conversations.json",encoding="utf-8") as f:
    conversations = json.load(f)
# print(conversations)
block_size=64
IGNORE_INDEX = -100

labels = []
input_ids = [] 

enc=tiktoken.get_encoding("gpt2")
formatted=[]
for item in conversations:
        
        prompt=f"user:{item['user']}\nassistant:"
        answer=f"{item['assistant']}\n"

        prompt_tokens = enc.encode(prompt)
        answer_tokens = enc.encode(answer)
        tokens=prompt_tokens + answer_tokens
        target=[IGNORE_INDEX]*len(prompt_tokens) + answer_tokens
        tokens=tokens[:block_size]
        target=target[:block_size]
        padding_length = block_size - len(tokens)
        tokens+=[enc.eot_token]*padding_length
        target+=[IGNORE_INDEX]*padding_length
        input_ids.append(tokens)
        labels.append(target)

input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)