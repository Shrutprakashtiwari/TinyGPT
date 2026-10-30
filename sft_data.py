import json
import tiktoken
import torch
import math
with open("dolly_conversations.json",encoding="utf-8") as f:
    conversations = json.load(f)
# print(conversations)
block_size=128
IGNORE_INDEX = -100

labels = []
input_ids = [] 
attention_masks=[]
enc=tiktoken.get_encoding("gpt2")
formatted=[]
for item in conversations:
        
        prompt=f"###User:{item['user']}\n###Assistant:"
        answer=f"{item['assistant']}\n"

        prompt_tokens = enc.encode(prompt)
        answer_tokens = enc.encode(answer)+[enc.eot_token]
        if len(prompt_tokens) + len(answer_tokens) > block_size:
            continue
        tokens=prompt_tokens + answer_tokens
        target=[IGNORE_INDEX]*len(prompt_tokens) + answer_tokens
        tokens=tokens[:block_size]
        target=target[:block_size]
        padding_length = block_size - len(tokens)
        attention_mask=[1]*len(tokens)
        attention_mask+=[0]*padding_length
        attention_masks.append(attention_mask)
        tokens+=[enc.eot_token]*padding_length
        target+=[IGNORE_INDEX]*padding_length
        
        input_ids.append(tokens)
        labels.append(target)
attention_masks = torch.tensor(attention_masks, dtype=torch.long)
input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# for i, item in enumerate(conversations):
#     if "python" in item["user"].lower():
#         print("USER:", item["user"])
#         print("EXPECTED:", item["assistant"])
#         print("TRAINED TOKENS:", (labels[i] != -100).sum().item())
batch_size = 8
num_batches=math.ceil(len(input_ids)*0.8/batch_size)
def get_batch(split):
    if split=="train":
         data=input_ids[:int(0.8*len(input_ids))]
         target=labels[:int(0.8*len(labels))]
         masks = attention_masks[:int(0.8*len(attention_masks))]
    else:
         data=input_ids[int(0.8*len(input_ids)):]
         target=labels[int(0.8*len(labels)):]
         masks = attention_masks[int(0.8*len(attention_masks)):]
    ix = torch.randint(len(data), (batch_size,))
    x = data[ix]
    y = target[ix]
    mask=masks[ix]
    return x, y,mask

valid_examples = (labels != IGNORE_INDEX).any(dim=1)

print("Total conversations:", len(labels))
print("Conversations with assistant targets:", valid_examples.sum().item())
print("Conversations with no assistant targets:", (~valid_examples).sum().item())
