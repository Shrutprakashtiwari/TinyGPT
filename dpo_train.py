import copy
from model import GPT
import torch
import torch.nn.functional as F
from preprocess import train_loader,preprocess,collate_fn
policy_model= GPT()
checkpoint= torch.load(
    "gpt_sft.pth",
    map_location="cpu",
    weights_only=True
)
policy_model.load_state_dict(checkpoint)
reference_model = copy.deepcopy(policy_model)
reference_model.eval()
# reference_model.load_state_dict(policy_model.state_dict())
for param in reference_model.parameters():
    param.requires_grad = False
policy_model.train()
def get_log_probs(model, input_ids,response_mask, attention_mask):
    logits, _,_=model(input_ids, attention_mask=attention_mask)
    logits=logits[:, :-1, :]
    targets=input_ids[:, 1:]
    response_mask=response_mask[:, 1:]
    log_probabs=F.log_softmax(logits, dim=-1)
    token_log_probs=log_probabs.gather( dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs=response_mask*token_log_probs
    sequence_log_probs=token_log_probs.sum(dim=-1)
    return sequence_log_probs

def compute_dpo_log_probs(batch):

    policy_chosen_logps = get_log_probs(
        policy_model,
        batch["chosen_ids"],
        batch["chosen_response_mask"],
        batch["chosen_attention_mask"]
    )

    policy_rejected_logps = get_log_probs(
        policy_model,
        batch["rejected_ids"],
        batch["rejected_response_mask"],
        batch["rejected_attention_mask"]
    )

    with torch.no_grad():

        reference_chosen_logps = get_log_probs(
            reference_model,
            batch["chosen_ids"],
            batch["chosen_response_mask"],
            batch["chosen_attention_mask"]
        )

        reference_rejected_logps = get_log_probs(
            reference_model,
            batch["rejected_ids"],
            batch["rejected_response_mask"],
            batch["rejected_attention_mask"]
        )

    return (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps
    )
beta=0.1
def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
    chosen_log_ratios = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratios = policy_rejected_logps - reference_rejected_logps
    logits=beta*(chosen_log_ratios-rejected_log_ratios)
    loss= -F.logsigmoid(logits).mean()
    return loss
import torch.optim as optim
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model=policy_model.to(device)
reference_model=reference_model.to(device)
# alpha=1e-5
learning_rate=1e-5
epochs=3
max_grad_norm=1.0
optimizer=optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=0.01)
for epoch in range(epochs):
    policy_model.train()
    reference_model.eval()
    total_loss=0.0
    for batch in train_loader:
        batch={
            key:value.to(device) for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        (policy_chosen_logps,
         policy_rejected_logps,
         reference_chosen_logps,
         reference_rejected_logps) = compute_dpo_log_probs(batch)
        loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}"
    )
    torch.save(
        policy_model.state_dict(),
        "gpt_dpo.pth")
    
