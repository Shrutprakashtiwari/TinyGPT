#For instruction-following data, retaining the end of a long prompt can be useful because it
#preserves the text closest to the assistant's response.
import tiktoken
import torch
import torch.nn.functional as F
from download_dpodataset import dataset

block_size = 128
batch_size = 8
tokenizer = tiktoken.get_encoding("gpt2")
eos_id = tokenizer.eot_token

train_data = dataset["train_prefs"]
test_data = dataset["test_prefs"]

def preprocess(data):
    processed_data = []

    for sample in data:
        prompt = f"### User:\n{sample['prompt']}\n\n### Assistant:\n"

        prompt_tokens = tokenizer.encode(prompt)
        chosen_tokens = tokenizer.encode(sample["chosen"][-1]["content"])
        rejected_tokens = tokenizer.encode(sample["rejected"][-1]["content"])

        chosen_tokens.append(eos_id)
        rejected_tokens.append(eos_id)

        prompt_tokens = prompt_tokens[-(block_size - 1):]
        response_budget = block_size - len(prompt_tokens)

        chosen_tokens = chosen_tokens[:response_budget]
        rejected_tokens = rejected_tokens[:response_budget]

        chosen_ids = prompt_tokens + chosen_tokens
        rejected_ids = prompt_tokens + rejected_tokens

        chosen_mask = [0] * len(prompt_tokens) + [1] * len(chosen_tokens)
        rejected_mask = [0] * len(prompt_tokens) + [1] * len(rejected_tokens)

        processed_data.append({
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask
        })

    return processed_data


train_data = preprocess(train_data)
test_data = preprocess(test_data)

import torch
from torch.utils.data import DataLoader

pad_id = tokenizer.eot_token

def collate_fn(batch):
    max_len = max(
        max(len(item["chosen_ids"]), len(item["rejected_ids"]))
        for item in batch
    )

    chosen_ids = []
    rejected_ids = []
    chosen_mask = []
    rejected_mask = []
    chosen_attention = []
    rejected_attention = []

    for item in batch:
        c = item["chosen_ids"]
        r = item["rejected_ids"]

        cm = item["chosen_mask"]
        rm = item["rejected_mask"]

        c_pad = max_len - len(c)
        r_pad = max_len - len(r)

        chosen_ids.append(c + [pad_id] * c_pad)
        rejected_ids.append(r + [pad_id] * r_pad)

        chosen_mask.append(cm + [0] * c_pad)
        rejected_mask.append(rm + [0] * r_pad)

        chosen_attention.append([1] * len(c) + [0] * c_pad)
        rejected_attention.append([1] * len(r) + [0] * r_pad)

    return {
        "chosen_ids": torch.tensor(chosen_ids),
        "rejected_ids": torch.tensor(rejected_ids),
        "chosen_mask": torch.tensor(chosen_mask),
        "rejected_mask": torch.tensor(rejected_mask),
        "chosen_attention": torch.tensor(chosen_attention),
        "rejected_attention": torch.tensor(rejected_attention)
    }
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
def get_log_probabs(model, input_ids,response_mask, attention_mask):
    logits, _,_=model(input_ids, attention_mask=attention_mask)
    logits=logits[:, :-1, :]
    targets=input_ids[:, 1:]
    response_mask=response_mask[:, 1:]
    log_probabs=F.log_softmax(logits, dim=-1)
    token_log_probs=log_probabs.gather(log_probabs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs=response_mask*token_log_probs
    sequence_log_probs=token_log_probs.sum(dim=-1)
    return sequence_log_probs
