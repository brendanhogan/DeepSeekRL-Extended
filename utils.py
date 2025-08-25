import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional
import pdb
import re
####################
## MISC FUNCTIONS ##
####################

def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [" ".join(re.split(r"\s+", line)).strip() for line in lines]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines



def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def write_generation_log(log_data: Dict[str, Any], log_file: str,dataset: str="gsm8k") -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: Dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, 'w') as f:
        # Write prompt section
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")
        f.write("#### ANS ####\n\n")
        f.write(str(log_data['prompt']['answer']) + "\n")
        if "function_signature" in log_data['prompt']:
            f.write("#### FUNCTION SIGNATURE ####\n\n")
            f.write(log_data['prompt']['function_signature'] + "\n\n")
        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")
            
            # Write individual scores
            if  "gsm" in dataset or "math500" in dataset:
                f.write(f"Correctness: {gen['scores']['correctness']}\n")
                f.write(f"Integer format: {gen['scores']['integer_format']}\n") 
                f.write(f"Strict format: {gen['scores']['strict_format']}\n")
                f.write(f"Soft format: {gen['scores']['soft_format']}\n")
                f.write(f"XML count: {gen['scores']['xml_count']}\n")
                f.write(f"Total reward: {gen['scores']['total_reward']}\n\n")
            elif dataset == "mbpp" or "leetcode" in dataset:
                f.write(f"Correctness: {gen['scores']['correctness']}\n")
                f.write(f"Syntax: {gen['scores']['syntax']}\n")
                f.write(f"Execution: {gen['scores']['execution']}\n")
                f.write(f"Format: {gen['scores']['format']}\n")
                f.write(f"Quality: {gen['scores']['quality']}\n\n")
                f.write(f"Total reward: {gen['scores']['total_reward']}\n\n")
            elif "reasoning_gym" in dataset:
                f.write(f"Correctness: {gen['scores']['correctness']}\n")
                f.write(f"Total reward: {gen['scores']['total_reward']}\n\n")
                f.write(f"XML count: {gen['scores']['xml_count']}\n")



####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)

    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
 
    return per_token_logps

def selective_log_softmax_with_multiple_indices(logits, indices):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.
    """
    per_token_logps = []
    for row_logits, row_indices in zip(logits, indices):
        row_logps = F.log_softmax(row_logits, dim=-1)
        row_per_token_logps = row_logps.gather(dim=-1, index=row_indices).squeeze(-1)
        per_token_logp = F.softmax(row_logps, dim=-1).gather(dim=-1, index=row_indices).squeeze(-1) + 1e-10
        per_token_logp = per_token_logp / per_token_logp.sum(dim=-1, keepdim=True)
        per_token_logp = per_token_logp * row_per_token_logps
        per_token_logp = per_token_logp.sum(dim=-1)
        per_token_logps.append(per_token_logp)
    per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def selective_log_softmax_with_multiple_indices_on_selected_tokens(logits, indices):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.
    """
    per_token_logps = []
    for row_logits, row_indices in zip(logits, indices):
        row_logps = F.log_softmax(row_logits, dim=-1)
        row_per_token_logps = row_logps.gather(dim=-1, index=row_indices.squeeze(-1)).squeeze(-1)
        per_token_logp = F.softmax(row_logps, dim=-1).gather(dim=-1, index=row_indices.squeeze(-1)).squeeze(-1) + 1e-10
        per_token_logp = per_token_logp / per_token_logp.sum(dim=-1, keepdim=True)
        per_token_logp = per_token_logp * row_per_token_logps
        per_token_logp = per_token_logp.sum(dim=-1)
        per_token_logps.append(per_token_logp)
    per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

def get_per_token_logps_with_embeddings(model, token_embeddings_list,input_ids, attention_mask, logits_to_keep,top_k=2,loss_on_all_tokens=False):
    """
    Generate per-token log probabilities from token embeddings for multiple chains.
    
    Args:
        model: The language model
        attention_mask: Attention mask tensor
        logits_to_keep: Number of logits to keep from the end
        token_embeddings_list: List of embedding tensors, one per chain
    
    Returns:
        per_token_logps: Log probabilities for each token in each chain
    """
    logits = model(inputs_embeds=token_embeddings_list,attention_mask=attention_mask,logits_to_keep=logits_to_keep + 1).logits
    # Exclude the last logit: it corresponds to the next token prediction
    logits = logits[:, :-1, :]  # (max_seq_len-1, vocab_size)
    input_ids = input_ids[:, -logits_to_keep:]
    all_top_k_tokens = torch.topk(logits, k=top_k, dim=-1).indices
    # For transformers<=4.48, logits_to_keep a  rgument isn't supported, so we drop logits ourselves
    logits = logits[:, -logits_to_keep:]
    # We need token IDs to compute log probabilities, but since we only have embeddings,
    # we'll need to get the token IDs from the model's tokenizer or find another way
    # For now, let's assume we need to return the logits and let the caller handle the token selection
    
    # Note: This function may need modification based on how the calling code expects to use it
    # since we don't have direct access to token IDs from embeddings
    if loss_on_all_tokens:
        return selective_log_softmax_with_multiple_indices(logits, all_top_k_tokens)
    else:
        return selective_log_softmax(logits, input_ids)
    # return selective_log_softmax(logits, input_ids)

def get_per_token_logps_with_embeddings_on_selected_tokens(model, token_embeddings_list,selected_tokens,input_ids, attention_mask, logits_to_keep):
    """
    Generate per-token log probabilities from token embeddings for multiple chains.
    
    Args:
        model: The language model
        attention_mask: Attention mask tensor
        logits_to_keep: Number of logits to keep from the end
        token_embeddings_list: List of embedding tensors, one per chain
    
    Returns:
        per_token_logps: Log probabilities for each token in each chain
    """
    logits = model(inputs_embeds=token_embeddings_list,attention_mask=attention_mask,logits_to_keep=logits_to_keep + 1).logits
    # Exclude the last logit: it corresponds to the next token prediction
    logits = logits[:, :-1, :]  # (max_seq_len-1, vocab_size)
    input_ids = input_ids[:, -logits_to_keep:]

    # For transformers<=4.48, logits_to_keep a  rgument isn't supported, so we drop logits ourselves
    logits = logits[:, -logits_to_keep:]
    # We need token IDs to compute log probabilities, but since we only have embeddings,
    # we'll need to get the token IDs from the model's tokenizer or find another way
    # For now, let's assume we need to return the logits and let the caller handle the token selection
    
    # Note: This function may need modification based on how the calling code expects to use it
    # since we don't have direct access to token IDs from embeddings
    return selective_log_softmax_with_multiple_indices_on_selected_tokens(logits, selected_tokens)
    # return selective_log_softmax(logits, input_ids)