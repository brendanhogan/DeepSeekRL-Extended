"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
import torch.nn.functional as F

import llms
import utils
import evaluator
import rldatasets
import pdb

@dataclass
class SoftToken:
    """Represents a soft token with weighted embeddings and probabilities"""
    token_ids: torch.Tensor  # [k] top-k token IDs
    probs: torch.Tensor      # [k] probabilities for each token
    embedding: torch.Tensor  # [hidden_dim] weighted embedding
    most_likely_id: int      # ID of most likely token (for readability)
    
    def to(self, device):
        """Move tensors to device"""
        return SoftToken(
            token_ids=self.token_ids.to(device),
            probs=self.probs.to(device), 
            embedding=self.embedding.to(device),
            most_likely_id=self.most_likely_id
        )

def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_loader: DataLoader for test set
        eval_class: Evaluator for computing rewards
        device: Device to run on
        args: Training arguments
        round_num: Current training round number
        
    Returns:
        total_scores: Dictionary of average metrics
        accuracy: Accuracy on test set
    """
    print("Running evaluation on test set...")
    
    # Track metrics across all test examples
    total_scores = defaultdict(float)
    num_examples = 0
    total_accuracy = 0.0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer, entry in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate completions using same function as training
            if args.normal_generation:
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
                    model, tokenizer, question, device, args
                )
            else:
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list, mixture_selected_tokens = generate_completions(
                    model, tokenizer, question, device, args
                )
            
            # Score completions using evaluator
            mock_prompts = [[{'content': question}]] * len(completions_text)
            mock_completions = [[{'content': completion}] for completion in completions_text]
            # Make answer array same length as completions
            answers = [answer] * len(completions_text)
            rewards_per_func, metrics = eval_class.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions, 
                answer=answers,
                device=device,
                entry=entry
            )
            
            # Track accuracy and accumulate metrics
            total_accuracy += metrics['accuracy']
                
            for k, v in metrics.items():
                if k == "correctness":
                    continue
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            
            # Log all completions
            for i, completion in enumerate(completions_text):
                f.write(f"Response {i+1}: {completion}\n")
            
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")
            
            # Log generation details if verbose
            if args.verbose and generation_log:
                f.write(f"Phase transitions: {generation_log.get('phase_transitions', [])}\n")
                f.write(f"Final sequence lengths: {generation_log.get('final_sequences', {}).get('sequence_lengths', [])}\n")


    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir,f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Accuracy: {accuracy:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, accuracy

@torch.no_grad()
def generate_with_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    device: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    """
    Generate completions by sampling tokens and feeding their embeddings back to the model.
    
    Returns:
        prompt_completion_ids: The generated sequences
        generation_log: Dictionary containing detailed generation information
    """
    num_chains = args.num_chains
    max_completion_length = args.max_completion_length
    temperature = args.temperature
    k = args.mixture_k
    
    # Get initial state from prompt
    outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True, return_dict=True)
    
    past_key_values = outputs.past_key_values
    
    # Get next token logits
    next_token_logits = outputs.logits[:, -1, :]
    token_embeddings_list = [[] for _ in range(num_chains)]
    for i in range(num_chains):
        token_embeddings_list[i].append(model.get_input_embeddings()(prompt_ids[i]))
    # Store generated tokens for each chain
    generated_sequences = [[] for _ in range(num_chains)]
    mixture_selected_tokens = [[] for _ in range(num_chains)]
    # Keep track of running chains
    model_device = next(model.parameters()).device
    running_chains = torch.ones(num_chains, dtype=torch.bool, device=model_device)
    
    # State for mixture vs normal generation
    is_mixture_phase = torch.ones(num_chains, dtype=torch.bool, device=model_device)
    try:
        think_end_token_ids = tokenizer.encode("</think>", add_special_tokens=False)
    except:
        print("Warning: Could not find </think> token. Mixture phase will run for all tokens.")
        think_end_token_ids = []

    # Initialize logging structures
    generation_log = {
        'generation_steps': [],  # List of step-by-step generation info
        'phase_transitions': [],  # When each chain switched from mixture to normal
        'think_end_token_ids': think_end_token_ids,
        'num_chains': num_chains,
        'max_completion_length': max_completion_length,
        'temperature': temperature
    }
    for step in range(max_completion_length):
        if not running_chains.any():
            break
            
        # Apply temperature
        scaled_logits = next_token_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Initialize step log
        step_log = {
            'step': step,
            'chains': []
        }
        
        # --- Mixture Path (for chains where is_mixture_phase is True) ---
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        normalized_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        top_k_embeddings = model.get_input_embeddings()(top_k_indices)
        ### Sampling Nucleus Wise 
        if "nucleus" in args.experiment_name:
            nucleus_sampling_threshold = 0.9
            normalized_probs = torch.cumsum(normalized_probs, dim=-1)
            normalized_probs = torch.where(normalized_probs > nucleus_sampling_threshold, top_k_probs, torch.zeros_like(normalized_probs))
            normalized_probs = normalized_probs / normalized_probs.sum(dim=-1, keepdim=True)
        ### Mixing Dirichlet Wise
        if "dirichlet" in args.experiment_name:
            normalized_probs = torch.distributions.Dirichlet(normalized_probs).sample()        
        if "different_tokens" in args.experiment_name:
            k_tokens = torch.multinomial(probs, num_samples=k)
            top_k_embeddings = model.get_input_embeddings()(k_tokens)
            top_k_probs = torch.gather(probs, dim=1, index=k_tokens)
            normalized_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True) 
        ### Mixing Element Wise Max Wise
        if "element_wise_max" in args.experiment_name:
            mixture_embedding = top_k_embeddings.max(dim=1, keepdim=True).values
        else: 
            mixture_embedding = (top_k_embeddings * normalized_probs.unsqueeze(-1)).sum(dim=1, keepdim=True)
        mixture_representative_idx = torch.multinomial(normalized_probs, num_samples=1)
        mixture_next_token = torch.gather(top_k_indices, 1, mixture_representative_idx)

        # --- Normal Sampling Path (for chains where is_mixture_phase is False) ---
        normal_next_token = torch.multinomial(probs, num_samples=1)
        normal_embedding = model.get_input_embeddings()(normal_next_token)

        # --- Select embedding and next token based on phase ---
        is_mixture_phase_b = is_mixture_phase.unsqueeze(1)
        
        token_embeddings = torch.where(
            is_mixture_phase_b.unsqueeze(2),
            mixture_embedding,
            normal_embedding
        )
        next_token = torch.where(
            is_mixture_phase_b,
            mixture_next_token,
            normal_next_token
        )
        
        # Store token and check for </think> to switch phase
        len_think_end = len(think_end_token_ids)
        for i in range(num_chains):
            if running_chains[i]:
                token_id = next_token[i].item()
                generated_sequences[i].append(token_id)
                if "different_tokens" in args.experiment_name:
                    if is_mixture_phase[i]:
                        mixture_token_ids = k_tokens[i]
                        mixture_selected_tokens[i].append(mixture_token_ids)
                    else:
                        mixture_selected_tokens[i].append(torch.full((args.mixture_k,), token_id, dtype=torch.long, device=model_device))

                # Log detailed information for this chain at this step
                chain_log = {
                    'chain_id': i,
                    'is_mixture_phase': is_mixture_phase[i].item(),
                    'selected_token_id': token_id,
                    'selected_token_text': tokenizer.decode([token_id]),
                    'is_running': True
                }
                if "different_tokens" in args.experiment_name:
                    chain_log.update({
                        'mixture_selected_tokens': mixture_selected_tokens[i][-1].tolist(),
                    })
                
                if is_mixture_phase[i]:
                    # Log mixture-specific information
                    chain_log.update({
                        'top_k_token_ids': top_k_indices[i].tolist(),
                        'top_k_token_texts': [tokenizer.decode([tid]) for tid in top_k_indices[i].tolist()],
                        'top_k_probs': top_k_probs[i].tolist(),
                        'normalized_mixture_weights': normalized_probs[i].tolist(),
                        'representative_idx': mixture_representative_idx[i].item()
                    })
                else:
                    # Log normal sampling information
                    chain_log.update({
                        'sampled_prob': probs[i, token_id].item()
                    })

                # Check if we should switch from mixture to normal generation
                if is_mixture_phase[i] and len_think_end > 0:
                    # Check for exact token sequence match
                    token_match = False
                    if len(generated_sequences[i]) >= len_think_end:
                        if generated_sequences[i][-len_think_end:] == think_end_token_ids:
                            token_match = True
                    
                    # Check for "</think>" text in generated content
                    text_match = False
                    if len(generated_sequences[i]) > 0:
                        generated_text = tokenizer.decode(generated_sequences[i], skip_special_tokens=True)
                        if "</think>" in generated_text:
                            text_match = True
                    
                    # Switch to normal generation if either condition is met
                    if token_match or text_match:
                        is_mixture_phase[i] = False
                        chain_log['phase_transition'] = True
                        generation_log['phase_transitions'].append({
                            'chain_id': i,
                            'step': step,
                            'token_position': len(generated_sequences[i]),
                            'trigger': 'token_match' if token_match else 'text_match'
                        })
                        if args.verbose:
                            trigger_type = "token sequence" if token_match else "text content"
                            print(f"Chain {i} finished mixture phase at step {step} (triggered by {trigger_type}).")
                
                step_log['chains'].append(chain_log)
                token_embeddings_list[i].append(token_embeddings[i])
            else:
                # Chain not running
                token_embeddings_list[i].append(torch.zeros(1, model.config.hidden_size, device=model_device,dtype=torch.bfloat16))
                step_log['chains'].append({
                    'chain_id': i,
                    'is_running': False
                })
        
        generation_log['generation_steps'].append(step_log)
        
        # Update running chains based on the representative token
        running_chains &= (next_token.squeeze(1) != tokenizer.eos_token_id)
        
        if not running_chains.any():
            break
            
        # Forward pass with the selected embedding
        

        outputs = model(
            inputs_embeds=token_embeddings,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    token_embeddings_list = [torch.cat(emb,dim=0).to(model.device) for emb in token_embeddings_list]
    token_embeddings_list = torch.stack(token_embeddings_list)

    # After the loop, pad the shorter sequences.
    max_len = 0
    if generated_sequences:
        max_len = max(len(tokens) for tokens in generated_sequences if tokens)

    if max_len > 0:
        padded_generated_ids = []
        for tokens in generated_sequences:
            padding_needed = max_len - len(tokens)
            padded_tokens = tokens + [tokenizer.pad_token_id] * padding_needed
            padded_generated_ids.append(padded_tokens)
        
        generated_ids = torch.tensor(padded_generated_ids, device=model_device, dtype=torch.long)
        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)
    
        # Pad mixture_selected_tokens
        for sel_list in mixture_selected_tokens:
            padding_needed = max_len - len(sel_list)
            for _ in range(padding_needed):
                sel_list.append(torch.full((args.mixture_k,), tokenizer.pad_token_id, dtype=torch.long, device=model_device))
    
        # Convert to tensors with proper shape for gather
        selected_tokens = [torch.stack(sel_list, dim=0).unsqueeze(-1) for sel_list in mixture_selected_tokens]
    else:
        # No tokens were generated
        prompt_completion_ids = prompt_ids
        selected_tokens = [torch.tensor([], dtype=torch.long, device=model_device) for _ in range(num_chains)]  # Empty for no generation

    # Add final summary to generation log
    generation_log['final_sequences'] = {
        'token_ids': [seq for seq in generated_sequences],
        'decoded_texts': [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences],
        'sequence_lengths': [len(seq) for seq in generated_sequences]
    }
    return prompt_completion_ids, generation_log, token_embeddings_list, selected_tokens

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str, dict | None, torch.Tensor | None, torch.Tensor | None]:

    """
    Generate multiple completion sequences for a given prompt using a language model.
    Routes to either normal generation or soft thinking generation based on args.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor (normal) or List[Union[Tensor, SoftToken]] (soft thinking)
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor (normal) or List[Union[Tensor, SoftToken]] (soft thinking)
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """
    if args.soft_thinking:
        return generate_completions_soft_thinking(model, tokenizer, question, device, args)
    else:
        return generate_completions_normal(model, tokenizer, question, device, args)


def generate_completions_normal(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences using normal (non-soft thinking) generation.
    This is the original generation function.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
        generation_log: Dictionary containing detailed generation information
    """
    # 1. Prepare prompting

    prompt = [
        {'role': 'system', 'content': args.system_prompt},
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': "<think>"},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, continue_final_message=True, tokenize=False)

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move tensors to model's device (handles multi-GPU setups)
    model_device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(model_device)
    prompt_mask = prompt_mask.to(model_device)

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )
    token_embeddings_list = None
    if args.normal_generation:
        # Generate completions
        prompt_completion_ids = model.generate(prompt_ids,attention_mask=prompt_mask,generation_config=generation_config)
    else:   
        # Generate completions
        prompt_completion_ids, generation_log, token_embeddings_list, selected_tokens = generate_with_embeddings(
            model,
            tokenizer,
            prompt_ids,
            prompt_mask,
            device,
            args
        )

    # Extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model_device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=model_device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    if args.normal_generation:
        return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text
    else: 
        return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list, selected_tokens

def score_completions(
    completions_text: list[str],
    question: str,
    answer: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    entry: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
            'answer': answer
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    mock_prompts = [[{'content': question}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]
    answers = [answer] * len(completions_text)
    
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=answers,
        device=device,  
        entry=entry
    )
    rewards = rewards_per_func.sum(dim=1)
    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data

def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel, 
    prompt_completion_ids: Union[torch.Tensor, List[List[Union[torch.Tensor, SoftToken]]]],
    prompt_ids: torch.Tensor,
    completion_ids: Union[torch.Tensor, List[List[Union[torch.Tensor, SoftToken]]]],
    attention_mask: torch.Tensor,
    completion_mask: Optional[torch.Tensor],
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    Routes to appropriate loss function based on whether using soft thinking.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs (tensor or list of mixed tokens)
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion (tensor or list of mixed tokens)
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion (for normal mode)
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """
    # Check if we're using soft thinking based on data types
    if isinstance(completion_ids, list) and len(completion_ids) > 0 and isinstance(completion_ids[0], list):
        # Soft thinking mode
        return compute_loss_soft_thinking(
            model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
            attention_mask, advantages, args
        )
    else:
        # Normal mode
        return compute_loss_normal(
            model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages, args
        )


def compute_loss_normal(
    model: PreTrainedModel,
    base_model: PreTrainedModel, 
    prompt_completion_ids: torch.Tensor,
    token_embeddings_list: torch.Tensor | None,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    selected_tokens: torch.Tensor | None,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model for normal (non-soft thinking) generation.
    This is the original compute_loss function.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        selected_tokens: Selected tokens for each chain
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)
    with torch.inference_mode():
        if args.normal_generation:
            ref_per_token_logps = utils.get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)
        elif "different_tokens" in args.experiment_name:
            ref_per_token_logps = utils.get_per_token_logps_with_embeddings_on_selected_tokens(base_model, token_embeddings_list, selected_tokens, prompt_completion_ids, attention_mask, logits_to_keep)
        else:
            ref_per_token_logps = utils.get_per_token_logps_with_embeddings(base_model, token_embeddings_list, prompt_completion_ids, attention_mask, logits_to_keep, top_k=args.mixture_k, loss_on_all_tokens=args.loss_on_all_tokens)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    if args.normal_generation:
        per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
    elif "different_tokens" in args.experiment_name:
        per_token_logps = utils.get_per_token_logps_with_embeddings_on_selected_tokens(model, token_embeddings_list, selected_tokens, input_ids, attention_mask, logits_to_keep)
    else:
        per_token_logps = utils.get_per_token_logps_with_embeddings(model, token_embeddings_list, input_ids, attention_mask, logits_to_keep, top_k=args.mixture_k, loss_on_all_tokens=args.loss_on_all_tokens)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics

def grpo_loss(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace,
        entry: dict | None = None
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """
    # Generate completions
    if args.normal_generation:
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
            model, tokenizer, question, device, args
        )
    else:
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list, selected_tokens = generate_completions(
            model, tokenizer, question, device, args
        )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, args, entry
    )

    # Add generation log to the log data
    log_data['generation_log'] = generation_log if not args.normal_generation else None

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file,args.dataset)
    
    # Also save detailed generation log as JSON for analysis
    if not args.normal_generation:
        detailed_log_file = os.path.join(training_log_dir, f'{round_num}_detailed_generation.json')
        
        with open(detailed_log_file, 'w') as f:
            json.dump(generation_log, f, indent=2)


    completion_mask = attention_mask[:, prompt_ids.size(1):]
    if args.normal_generation:
        loss, loss_metrics = compute_loss(
            model, base_model, prompt_completion_ids, None, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages,None, args
        )
    else:
        loss, loss_metrics = compute_loss(
            model, base_model, prompt_completion_ids, token_embeddings_list, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages, selected_tokens, args

        )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

def generate_completions_soft_thinking(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[List[Union[torch.Tensor, SoftToken]], torch.Tensor, List[Union[torch.Tensor, SoftToken]], torch.Tensor, list[str], str]:
    """
    Generate completions using soft thinking - weighted mixtures of top-k token embeddings.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_sequence: List of tokens (hard tokens + soft tokens)
        prompt_ids: Tensor containing just the prompt token IDs  
        completion_sequence: List of completion tokens (mix of hard and soft)
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts (using most likely tokens)
        prompt_text: The full formatted prompt text
    """
    # 1. Prepare prompting (same as before)
    # Import the system prompt
    from rldatasets import SYSTEM_PROMPT
    prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Debug: Check tokenizer output
    if args.verbose:
        print(f"Tokenizer output:")
        print(f"  prompt_ids shape: {prompt_ids.shape}")
        print(f"  prompt_mask shape: {prompt_mask.shape}")

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Debug: Check after truncation
    if args.verbose:
        print(f"After truncation:")
        print(f"  prompt_ids shape: {prompt_ids.shape}")
        print(f"  prompt_mask shape: {prompt_mask.shape}")
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)
    
    # Debug: Check after repeat
    if args.verbose:
        print(f"After repeat (num_chains={args.num_chains}):")
        print(f"  prompt_ids shape: {prompt_ids.shape}")
        print(f"  prompt_mask shape: {prompt_mask.shape}")

    # Move to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    
    # Get model's embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Get special token IDs
    reasoning_end_ids = tokenizer.encode("</reasoning>", add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    
    # Initialize for generation
    batch_size = args.num_chains
    current_ids = prompt_ids  # [batch_size, prompt_len]
    current_attention_mask = prompt_mask  # [batch_size, prompt_len]
    
    # Debug: Check initial dimensions
    if args.verbose:
        print(f"Initialized soft thinking generation:")
        print(f"  batch_size: {batch_size}")
        print(f"  prompt_ids shape: {prompt_ids.shape}")
        print(f"  prompt_mask shape: {prompt_mask.shape}")
    
    # Track completion sequences for each chain
    completion_sequences = [[] for _ in range(batch_size)]
    most_likely_sequences = [[] for _ in range(batch_size)]  # For decoding
    
    # Generation loop
    for step in range(args.max_completion_length):
        # Get model outputs for current sequence
        with torch.no_grad():
            # For the current step, we need to handle mixed hard/soft tokens
            if step == 0:
                # First step - use hard tokens (prompt)
                outputs = model(input_ids=current_ids, attention_mask=current_attention_mask)
            else:
                # For subsequent steps, we need to construct inputs with soft embeddings
                # Process all sequences in parallel for proper batching
                inputs_embeds = []
                attention_masks = []
                
                for batch_idx in range(batch_size):
                    # Get embeddings for this batch item
                    prompt_embeds = embed_layer(current_ids[batch_idx:batch_idx+1])  # [1, prompt_len, hidden]
                    
                    # Add soft token embeddings from previous steps
                    batch_embeds = [prompt_embeds.squeeze(0)]  # [prompt_len, hidden]
                    for soft_token in completion_sequences[batch_idx]:
                        if isinstance(soft_token, SoftToken):
                            # Soft token embedding
                            soft_embed = soft_token.embedding.unsqueeze(0)  # [1, hidden]
                            batch_embeds.append(soft_embed)
                        else:
                            # Hard token - ensure consistent tensor handling
                            if isinstance(soft_token, torch.Tensor):
                                if soft_token.dim() == 0:
                                    # Scalar tensor
                                    token_id = soft_token.unsqueeze(0)  # [1]
                                elif soft_token.dim() == 1:
                                    # Already 1D
                                    token_id = soft_token  # [1] or [n]
                                else:
                                    # Take first element if multi-dimensional
                                    token_id = soft_token.flatten()[:1]  # [1]
                            else:
                                # Convert to tensor if needed
                                token_id = torch.tensor([soft_token], device=device)
                            
                            hard_embed = embed_layer(token_id.unsqueeze(0))  # [1, len, hidden]
                            # Squeeze to get [len, hidden], then take first token [1, hidden]
                            hard_embed = hard_embed.squeeze(0)  # [len, hidden]
                            if hard_embed.dim() == 2 and hard_embed.size(0) > 1:
                                hard_embed = hard_embed[:1]  # Take first token: [1, hidden]
                            elif hard_embed.dim() == 1:
                                hard_embed = hard_embed.unsqueeze(0)  # [1, hidden]
                            batch_embeds.append(hard_embed)
                    
                    full_embeds = torch.cat(batch_embeds, dim=0)  # [total_len, hidden]
                    inputs_embeds.append(full_embeds)

                # Pad to same length and stack - this is crucial for proper batching
                max_len = max(embeds.size(0) for embeds in inputs_embeds)
                padded_embeds = []
                
                for embeds in inputs_embeds:
                    pad_len = max_len - embeds.size(0)
                    if pad_len > 0:
                        pad_embeds = torch.zeros(pad_len, embeds.size(1), device=device, dtype=embeds.dtype)
                        embeds = torch.cat([pad_embeds, embeds], dim=0)
                    padded_embeds.append(embeds)
                    
                    # Create attention mask
                    attention_mask = torch.ones(max_len, device=device)
                    if pad_len > 0:
                        attention_mask[:pad_len] = 0
                    attention_masks.append(attention_mask)
                
                # Stack to create proper batch dimensions
                inputs_embeds = torch.stack(padded_embeds)  # [batch_size, max_len, hidden]
                current_attention_mask = torch.stack(attention_masks)  # [batch_size, max_len]
                
                # Verify batch dimensions before forward pass
                if inputs_embeds.size(0) != batch_size:
                    print(f"Warning: inputs_embeds batch size {inputs_embeds.size(0)} != expected {batch_size}")
                
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=current_attention_mask)
        
        # Get logits for next token
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Debug: Check dimensions
        if step < 2 and args.verbose:
            print(f"Step {step}: outputs.logits shape: {outputs.logits.shape}")
            print(f"Step {step}: logits shape: {logits.shape}")
        
        # Apply temperature
        logits = logits / args.temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]
        
        # Debug: Check probability dimensions
        if step < 2 and args.verbose:
            print(f"Step {step}: probs shape: {probs.shape}, expected batch_size: {batch_size}")
        
        # Check if we should exit soft thinking mode for any sequence
        # (when </reasoning> is the most likely token)
        most_likely_tokens = torch.argmax(probs, dim=-1)  # [batch_size]
        
        # Generate next tokens for each sequence
        next_tokens = []
        

        for batch_idx in range(batch_size):
            batch_probs = probs[batch_idx]
            most_likely_id = most_likely_tokens[batch_idx].item()
            
            # Check if we should exit soft thinking - either EOS or if last tokens match </reasoning>
            sequence_so_far = most_likely_sequences[batch_idx]
            should_exit = (most_likely_id == eos_token_id or 
                          (len(sequence_so_far) >= len(reasoning_end_ids) and 
                           sequence_so_far[-len(reasoning_end_ids):] == reasoning_end_ids))
            
            if should_exit:
                # Switch to hard token generation
                sampled_id = torch.multinomial(batch_probs, 1).item()
                # Store as scalar tensor for consistency
                next_tokens.append(torch.tensor(sampled_id, device=device))
                most_likely_sequences[batch_idx].append(sampled_id)

            else:
                # Continue with soft thinking - create mixed token
                # Sample k tokens from distribution for diversity
                top_ids = torch.multinomial(batch_probs, args.soft_thinking_k)
                top_probs = batch_probs[top_ids]
                
                # Normalize probabilities
                top_probs = top_probs / top_probs.sum()

                
                # Get embeddings for top-k tokens
                top_embeddings = embed_layer(top_ids)  # [k, hidden_dim]
                
                # Create weighted embedding
                weighted_embedding = torch.sum(top_probs.unsqueeze(-1) * top_embeddings, dim=0)  # [hidden_dim]
                
                # Create soft token
                soft_token = SoftToken(
                    token_ids=top_ids,
                    probs=top_probs,
                    embedding=weighted_embedding,
                    most_likely_id=most_likely_id
                )
                
                next_tokens.append(soft_token)
                most_likely_sequences[batch_idx].append(most_likely_id)
                if args.verbose and step < 3 and batch_idx == 0:  # Only print for first chain and first few steps
                    decoded_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
                    print(f"Step {step}, Chain {batch_idx}: Soft thinking, top-{args.soft_thinking_k} tokens: {decoded_tokens}, probs: {top_probs.tolist()}")
        
        # Add tokens to completion sequences
        for batch_idx, token in enumerate(next_tokens):
            completion_sequences[batch_idx].append(token)
        
        # Check for early stopping (all sequences hit EOS or reasoning end)
        all_finished = True
        for batch_idx in range(batch_size):
            sequence_so_far = most_likely_sequences[batch_idx]
            if len(sequence_so_far) == 0:
                all_finished = False
                break
            
            last_token_id = sequence_so_far[-1]
            sequence_ended = (last_token_id == eos_token_id or 
                            (len(sequence_so_far) >= len(reasoning_end_ids) and 
                             sequence_so_far[-len(reasoning_end_ids):] == reasoning_end_ids))
            
            if not sequence_ended:
                all_finished = False
                break
        
        if all_finished:
            break
    
    # Convert most likely sequences to text for evaluation
    completions_text = []
    for seq in most_likely_sequences:
        if seq:  # Check if sequence is not empty
            try:
                text = tokenizer.decode(seq, skip_special_tokens=True)
                completions_text.append(text)
            except:
                # Fallback in case of decoding issues
                completions_text.append("")
        else:
            completions_text.append("")
    
    # Create combined sequences (prompt + completion)
    prompt_completion_sequences = []
    for batch_idx in range(batch_size):
        # Convert prompt to list of hard tokens
        prompt_tokens = [prompt_ids[batch_idx, i] for i in range(prompt_ids.size(1))]
        combined_seq = prompt_tokens + completion_sequences[batch_idx]
        prompt_completion_sequences.append(combined_seq)
    
    # Create attention masks (simplified - assume all tokens are attended to)
    max_total_len = max(len(seq) for seq in prompt_completion_sequences)
    attention_mask = torch.ones(batch_size, max_total_len, device=device)
    
    return prompt_completion_sequences, prompt_ids, completion_sequences, attention_mask, completions_text, prompt_text

def compute_loss_soft_thinking(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    prompt_completion_sequences: List[List[Union[torch.Tensor, SoftToken]]],
    prompt_ids: torch.Tensor,
    completion_sequences: List[List[Union[torch.Tensor, SoftToken]]],
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss for soft thinking generation.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_sequences: List of mixed token sequences (prompt + completion)
        prompt_ids: Tensor containing just the prompt token IDs
        completion_sequences: List of completion sequences with mixed tokens
        attention_mask: Attention mask tensor
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """
    device = prompt_ids.device
    batch_size = len(completion_sequences)
    
    # We need to compute loss for each sequence individually due to mixed tokens
    total_loss = 0.0
    total_kl = 0.0
    total_response_length = 0.0
    
    embed_layer = model.get_input_embeddings()
    
    for batch_idx in range(batch_size):
        completion_seq = completion_sequences[batch_idx]
        if not completion_seq:  # Skip empty sequences
            continue
            
        # Build input embeddings for this sequence
        prompt_embeds = embed_layer(prompt_ids[batch_idx:batch_idx+1])  # [1, prompt_len, hidden]
        sequence_embeds = [prompt_embeds.squeeze(0)]  # [prompt_len, hidden]
        
        # Track positions of soft vs hard tokens for loss computation
        completion_positions = []
        completion_targets = []  # For hard tokens
        completion_soft_targets = []  # For soft tokens
        
        for pos, token in enumerate(completion_seq):
            if isinstance(token, SoftToken):
                sequence_embeds.append(token.embedding.unsqueeze(0))  # [1, hidden]
                completion_positions.append(len(sequence_embeds) - 1)
                completion_targets.append(None)  # No single target for soft token
                completion_soft_targets.append(token)
            else:
                # Hard token
                hard_embed = embed_layer(token.unsqueeze(0).unsqueeze(0))  # [1, 1, hidden]
                sequence_embeds.append(hard_embed.squeeze(0))  # [1, hidden]
                completion_positions.append(len(sequence_embeds) - 1)
                completion_targets.append(token.item())
                completion_soft_targets.append(None)
        
        if not completion_positions:  # No completion tokens
            continue
            
        # Concatenate embeddings and get model outputs
        full_embeds = torch.cat(sequence_embeds, dim=0).unsqueeze(0)  # [1, total_len, hidden]
        seq_attention_mask = torch.ones(1, full_embeds.size(1), device=device)
        
        # Get model outputs
        outputs = model(inputs_embeds=full_embeds, attention_mask=seq_attention_mask)
        logits = outputs.logits.squeeze(0)  # [total_len, vocab_size]
        
        # Get reference model outputs for KL computation
        with torch.inference_mode():
            ref_outputs = base_model(inputs_embeds=full_embeds, attention_mask=seq_attention_mask)
            ref_logits = ref_outputs.logits.squeeze(0)  # [total_len, vocab_size]
        
        # Compute per-token losses and KL divergences
        token_losses = []
        token_kls = []
        
        for i, pos in enumerate(completion_positions):
            if pos >= logits.size(0) - 1:  # Skip if position is out of bounds
                continue
                
            # Get logits for next token prediction (logits at position pos predict token at pos+1)
            current_logits = logits[pos - 1]  # [vocab_size] - predict token at current position
            ref_current_logits = ref_logits[pos - 1]  # [vocab_size]
            
            # Apply temperature
            current_logits = current_logits / args.temperature
            ref_current_logits = ref_current_logits / args.temperature
            
            # Get probabilities
            current_probs = F.softmax(current_logits, dim=-1)
            ref_current_probs = F.softmax(ref_current_logits, dim=-1)
            
            # Compute loss based on token type
            if completion_targets[i] is not None:
                # Hard token - standard cross-entropy loss
                target_id = completion_targets[i]
                per_token_logp = F.log_softmax(current_logits, dim=-1)[target_id]
                ref_per_token_logp = F.log_softmax(ref_current_logits, dim=-1)[target_id]
                
                # KL divergence for this token
                kl = torch.exp(ref_per_token_logp - per_token_logp) - (ref_per_token_logp - per_token_logp) - 1
                
                # GRPO loss
                token_loss = torch.exp(per_token_logp - per_token_logp.detach()) * advantages[batch_idx]
                token_loss = -(token_loss - args.kl_weight_beta * kl)
                
            else:
                # Soft token - weighted loss over top-k tokens
                soft_token = completion_soft_targets[i]
                token_ids = soft_token.token_ids
                token_probs = soft_token.probs
                
                # Compute weighted loss
                weighted_loss = 0.0
                weighted_kl = 0.0
                
                for j, (token_id, prob) in enumerate(zip(token_ids, token_probs)):
                    # Log probability for this token
                    per_token_logp = F.log_softmax(current_logits, dim=-1)[token_id]
                    ref_per_token_logp = F.log_softmax(ref_current_logits, dim=-1)[token_id]
                    
                    # KL divergence for this token
                    kl = torch.exp(ref_per_token_logp - per_token_logp) - (ref_per_token_logp - per_token_logp) - 1
                    
                    # GRPO loss for this token
                    token_loss = torch.exp(per_token_logp - per_token_logp.detach()) * advantages[batch_idx]
                    token_loss = -(token_loss - args.kl_weight_beta * kl)
                    
                    # Weight by probability
                    weighted_loss += prob * token_loss
                    weighted_kl += prob * kl
                
                token_loss = weighted_loss
                kl = weighted_kl
            
            token_losses.append(token_loss)
            token_kls.append(kl)
        
        # Aggregate losses for this sequence
        if token_losses:
            sequence_loss = torch.stack(token_losses).mean()
            sequence_kl = torch.stack(token_kls).mean()
            
            total_loss += sequence_loss
            total_kl += sequence_kl
            total_response_length += len(token_losses)
    
    # Average across batch
    if batch_size > 0:
        loss = total_loss / batch_size
        mean_kl = total_kl / batch_size
        mean_response_length = total_response_length / batch_size
    else:
        loss = torch.tensor(0.0, device=device)
        mean_kl = torch.tensor(0.0, device=device)
        mean_response_length = 0.0
    
    # Additional metrics
    metrics = {
        "response_length": mean_response_length,
        "kl": mean_kl.item() if isinstance(mean_kl, torch.Tensor) else mean_kl
    }
    
    return loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    default_system_prompt =  """
Respond in the following format only:
<think>
...
</think>
<answer>
...
</answer>
"""
    default_system_prompt =  """You will be given a question that involves reasoning. You should first think about different approaches to solve the question, then reason about those approaches step by step and finally provide your answer.
    Think in roughly 200-400 words.
            It is very important that you put your reasoning process inside <think> tags and your final answer inside <answer> tags, like this:
            <think>
            Your step-by-step reasoning process here
            </think>
            <answer>
            Your final answer here
            </answer>
            All of your returned text should either be in the <think> or <answer> tags - no text outside! Start each answer by immediately starting with <think>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!"""
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use for training (e.g., gsm8k, math500)")
    parser.add_argument("--evaluator", type=str, default="gsm8k", help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output_new", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=100, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=50, help="Number of iterations for evaluation")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for warmup")
    parser.add_argument("--update_ref_model", action="store_true", help="Whether to update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="How often to update reference model")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Alpha parameter for reference model mixup")


    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.1, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")
    parser.add_argument("--mixture_k", type=int, default=2, help="Mixture k")
    parser.add_argument("--loss_on_all_tokens", type=int, default=0, help="Loss on all tokens")
    # Flash attention parameters
    parser.add_argument("--use_flash_attention", type=int, default=1, help="Use flash attention")
    parser.add_argument("--system_prompt", type=str, default=default_system_prompt, help="System prompt")   
    
    parser.add_argument("--normal_generation", type=int, default=0, help="Use normal generation")
    parser.add_argument("--experiment_name", type=str, default="different_tokens_1.5b_reasoning_gym_evaluator", help="Experiment name")
    parser.add_argument("--checkpoint_frequency", type=int, default=200, help="Save model checkpoint every N iterations")

    # Soft thinking parameters
    parser.add_argument("--soft_thinking", action="store_true", help="Enable soft thinking mode with weighted token embeddings")
    parser.add_argument("--soft_thinking_k", type=int, default=2, help="Number of top-k tokens to mix in soft thinking")
    parser.add_argument("--soft_thinking_temperature", type=float, default=1.0, help="Temperature for soft thinking probability mixing")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    
    # Seed everything 
    utils.seed_everything(args.seed)

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')
    
    # Print mode information
    if args.soft_thinking:
        print(f"🧠 SOFT THINKING MODE ENABLED")
        print(f"   - Top-k tokens: {args.soft_thinking_k}")
        print(f"   - Soft thinking temperature: {args.soft_thinking_temperature}")
        print(f"   - Will exit soft thinking when </reasoning> is most likely")
    else:
        print("🔢 NORMAL GENERATION MODE") 

    ###############################
    ## Main Experiment settings ##
    ###############################

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device,use_flash_attention=args.use_flash_attention)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device,use_flash_attention=args.use_flash_attention)
    
    # Store original parameters for hard reset
    original_base_model_params = {}
    for name, param in base_model.named_parameters():
        original_base_model_params[name] = param.data.clone()

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.dataset)

    ###############################


    # Setup logging 
    output_dir = args.output_dir #if args.normal_generation else os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    eval_log_dir = os.path.join(output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)


    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)


    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
        
        # Evaluate on test set every so often 
        if (round_num + 1) % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Hard reset reference model to original parameters every 1000 steps
        if (round_num + 1) % 1000 == 0:
            print(f"Hard resetting reference model to original parameters at step {round_num + 1}")
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if name in original_base_model_params:
                        param.data.copy_(original_base_model_params[name])

        # Get next question
        try:
            question, answer, entry = next(train_loader)
        except StopIteration:
            print("Reached end of training set. Restarting from beginning.")
            train_loader.reset()
            question, answer, entry = next(train_loader)
            print(f"Restarted from beginning. Question: {question}")

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(model, base_model, tokenizer, question, answer, eval_class, device, round_num, train_log_dir, args, entry)
        
        # Gradient accumulation
        total_loss = total_loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
       
        if (round_num + 1) % args.checkpoint_frequency == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{round_num+1}')
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

   