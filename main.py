"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig

import llms
import utils
import evaluator
import rldatasets
from rldatasets import SYSTEM_PROMPT

class HiddenStateInjector(torch.nn.Module):
    """
    A simple network that learns to transform the LLM's hidden state into 
    a virtual token that helps with task performance.
    
    This network takes the final hidden state after processing the prompt,
    transforms it through a few layers, and outputs a new hidden state that
    gets injected as the "next token" before normal generation continues.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Simple autoencoder-like architecture
        # Compress to smaller representation then back to original size
        compress_size = hidden_size // 4
        
        self.transform_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, compress_size),
            torch.nn.ReLU(),
            torch.nn.Linear(compress_size, hidden_size),
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_size, hidden_size)
        )
        
        # Add RMS normalization to the output
        self.rms_norm = torch.nn.RMSNorm(hidden_size)
        
        # Initialize as identity mapping to start conservatively
        # self._initialize_as_identity()
    
    def _initialize_as_identity(self):
        """Initialize the network to approximate identity mapping"""
        # Zero out all layers except make the final layer close to identity
        for layer in self.transform_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        # Make the final linear layer start as identity (as much as possible)
        final_layer = self.transform_network[-1]
        torch.nn.init.eye_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Transform the input hidden state into a new representation
        
        Args:
            hidden_state: Tensor of shape (batch_size, hidden_size)
            
        Returns:
            transformed_state: Tensor of shape (batch_size, hidden_size)
        """
        transformed = self.transform_network(hidden_state)
        # Apply RMS normalization to the output
        return self.rms_norm(transformed)


def generate_completions_with_injection(
    model: PreTrainedModel,
    injector: HiddenStateInjector,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate completions using our hidden state injection method.
    
    This function:
    1. Processes the prompt normally to get hidden states
    2. Takes the final hidden state and transforms it via our injector network
    3. Injects this as a "virtual token" at the embedding level
    4. Continues generation normally from there
    
    Args:
        model: The language model (frozen during training)
        injector: Our learnable hidden state transformation network
        tokenizer: Tokenizer for the model
        question: The input question/prompt
        device: Device to run on
        args: Training arguments
        
    Returns:
        Same format as original generate_completions function
    """
    # Step 1: Prepare the prompt exactly as before
    prompt = [
        {'role': 'system', 'content': train_loader.system_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate and repeat for parallel generation
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Step 2: Get the final hidden state after processing the prompt
    with torch.no_grad():
        prompt_outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask, output_hidden_states=True)
        final_hidden_states = prompt_outputs.hidden_states[-1]  # Last layer
        last_token_hidden = final_hidden_states[:, -1, :].detach()  # Final token of each sequence


        # Step 3: Transform the hidden state using our injector network
        virtual_token_embedding = injector(last_token_hidden).unsqueeze(1)  # Add sequence dimension
        
    # Debug: check if virtual embedding requires gradients
    # print(f"Virtual embedding requires_grad: {virtual_token_embedding.requires_grad}")
    # print(f"Final hidden requires_grad: {last_token_hidden.requires_grad}")
    
    # Step 4: Get original prompt embeddings and concatenate virtual token
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(prompt_ids)
    
    # Concatenate the virtual token embedding
    extended_embeddings = torch.cat([prompt_embeddings, virtual_token_embedding], dim=1)
    extended_attention_mask = torch.cat([prompt_mask, torch.ones(prompt_mask.shape[0], 1, device=device)], dim=1)
    

    # Step 5: Generate from the extended context (prompt + virtual token)
    # Since we can't represent the virtual embedding as a token ID, we'll need a slightly different approach
    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Generate using the extended embeddings
    with torch.no_grad():
        generation_output = model.generate(
            inputs_embeds=extended_embeddings,
            attention_mask=extended_attention_mask,
            generation_config=generation_config
        )
    
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         inputs_embeds=prompt_embeddings,
    #         attention_mask=prompt_mask,
    #         generation_config=generation_config
    #     )
    

    # Extract the generated token sequences
    generated_sequences = generation_output.sequences
    
    # The generated sequences will include the original prompt tokens + new completion tokens
    # We need to extract just the completion tokens
    prompt_length = prompt_ids.shape[1]
    completion_ids = generated_sequences
    
    # Create prompt_completion_ids by concatenating original prompt with completions
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # Create attention masks
    completion_length = completion_ids.shape[1]
    completion_mask = torch.ones(prompt_ids.shape[0], completion_length, device=device)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    
    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    full_text = tokenizer.batch_decode(prompt_completion_ids, skip_special_tokens=False)

    # with open('test.txt', 'w') as f:
    #     for i, text in enumerate(full_text, 1):
    #         # Get the first token after our soft input as the decoded soft token
    #         first_completion_token = tokenizer.decode(completion_ids[i-1, 0], skip_special_tokens=False)
    #         f.write(f"decoded soft token: {first_completion_token}\n")
    #         f.write(f"Text {i}:\n{text}\n\n")
    # aaaa
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text


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
        for question, answer in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate completions using same function as training
            _, _, _, _, completions_text, _ = generate_completions(
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
                device=device
            )
            
            # Track accuracy and accumulate metrics
            total_accuracy += metrics['accuracy']
                
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {completions_text[0]}\n") # Log first completion
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")


    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
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

def eval_on_test_set_with_injection(
    model: PreTrainedModel,
    injector: HiddenStateInjector,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set using hidden state injection.
    
    Args:
        model: The model to evaluate
        injector: The hidden state injector network
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
    print("Running evaluation on test set with injection...")
    
    # Track metrics across all test examples
    total_scores = defaultdict(float)
    num_examples = 0
    total_accuracy = 0.0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'eval_metrics_injection_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer in tqdm(test_loader, desc="Evaluating on test set with injection"):
            # Generate completions using injection method
            _, _, _, _, completions_text, _ = generate_completions_with_injection(
                model, injector, tokenizer, question, device, args
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
                device=device
            )
            
            # Track accuracy and accumulate metrics
            total_accuracy += metrics['accuracy']
                
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {completions_text[0]}\n") # Log first completion
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")


    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_injection_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results (with injection):")
        print("-" * 20)
        print(f"Accuracy: {accuracy:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, accuracy

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
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
    """
    # 1. Prepare prompting
    prompt = [
        {'role': 'system', 'content': train_loader.system_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move tensors to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    # Generate completions
    prompt_completion_ids = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config
    )

    # Extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text
    
def score_completions(
    completions_text: list[str],
    question: str,
    answer: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
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
        device=device
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
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

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
        args: argparse.Namespace
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
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, question, device, args
    )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

def grpo_loss_with_injection(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        injector: HiddenStateInjector,
        base_injector: HiddenStateInjector,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss using hidden state injection.
    
    We generate tokens using the injector, then compute the policy gradient loss
    using advantages. For reference probabilities, we use the base_injector.
    """
    # Generate completions using our current injector
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions_with_injection(
        model, injector, tokenizer, question, device, args
    )



    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Get current policy log probabilities for the generated tokens
    # We need to recompute the forward pass with our injector to get logits
    current_logps = get_injection_logps(model, injector, prompt_ids, completion_ids, tokenizer, device, args)
    

    # Compute GRPO loss using the log probabilities
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    
    # Policy gradient loss with advantages
    per_token_loss = torch.exp(current_logps - current_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    response_length = completion_mask.sum(1).float().mean().item()
    
    loss_metrics = {
        "response_length": response_length,
    }
    
    metrics.update(loss_metrics)
    return loss, metrics

def get_injection_logps(model, injector, prompt_ids, completion_ids, tokenizer, device, args):
    """Get log probabilities for completion tokens using injection method."""
    # Recreate the injection context
    prompt = [
        {'role': 'system', 'content': train_loader.system_prompt},
        {'role': 'user', 'content': "dummy"}  # We'll use prompt_ids directly
    ]
    
    # Get hidden state and inject virtual token - NEED GRADIENTS HERE!
    with torch.no_grad():
        prompt_outputs = model(input_ids=prompt_ids, output_hidden_states=True)
        final_hidden = prompt_outputs.hidden_states[-1][:, -1, :].detach()  # Detach from model
    virtual_embedding = injector(final_hidden).unsqueeze(1)  # This needs gradients!
    # print(virtual_embedding) 
    # loss = 1 - virtual_embedding
    # loss.backward() 
    
    
    # Debug: check if virtual embedding requires gradients
    # print(f"Virtual embedding requires_grad: {virtual_embedding.requires_grad}")
    # print(f"Final hidden requires_grad: {final_hidden.requires_grad}")
    
    # Create extended embeddings with virtual token
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(prompt_ids)
    extended_embeddings = torch.cat([prompt_embeddings, virtual_embedding], dim=1)
    
    # Create full sequence embeddings (prompt + virtual + completion)
    completion_embeddings = embedding_layer(completion_ids)
    full_embeddings = torch.cat([extended_embeddings, completion_embeddings], dim=1)
    
    # Create attention mask
    prompt_mask = torch.ones_like(prompt_ids)
    virtual_mask = torch.ones(prompt_ids.shape[0], 1, device=device)
    completion_mask = torch.ones_like(completion_ids)
    full_attention_mask = torch.cat([prompt_mask, virtual_mask, completion_mask], dim=1)
    
    # Forward pass to get logits
    outputs = model(inputs_embeds=full_embeddings, attention_mask=full_attention_mask)
    logits = outputs.logits
    
    # Extract logits for completion tokens and compute log probabilities
    completion_logits = logits[:, prompt_ids.shape[1] + 1: prompt_ids.shape[1] + 1 + completion_ids.shape[1], :]
    completion_logps = torch.log_softmax(completion_logits, dim=-1)
    
    # Get log probabilities for the actual completion tokens
    per_token_logps = torch.gather(completion_logps, -1, completion_ids.unsqueeze(-1)).squeeze(-1)
    
    return per_token_logps

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="gsm8k", help="Evaluator to use for scoring")
    
    # Experimental setting
    parser.add_argument("--use_hidden_injection", action="store_true", help="Use hidden state injection instead of standard GRPO")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=100, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=20, help="Number of iterations for evaluation")

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
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

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

    ###############################
    ## Main Experiment settings ##
    ###############################

    ## Load the base language models
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    # base_model, _ = llms.get_llm_tokenizer(args.model_name, device)
    base_model = None 
    
    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)
    
    if args.use_hidden_injection:
        
        # Freeze the language models 
        for param in model.parameters():
            param.requires_grad = False
        
        # Create our hidden state injector networks
        hidden_size = model.config.hidden_size
        injector = HiddenStateInjector(hidden_size).to(device)
        base_injector = None
        
        # Match the model's dtype (likely BFloat16)
        model_dtype = next(model.parameters()).dtype
        injector = injector.to(dtype=model_dtype)
        

        # Setup optimizer for the injector network only
        optimizer = torch.optim.AdamW(
            injector.parameters(), 
            lr=0.001
        )
        
    else:

        # Setup optimizer for the full model (standard GRPO)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=1e-8
        )

    ###############################

    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        # if step < warmup_steps:
        #     return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Begin training!
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    
    for round_num in tqdm(range(args.num_train_iters), desc=f"Training"):
    
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0:
            print(f"Running evaluation at step {round_num}...")
            if not args.use_hidden_injection:
                # Standard evaluation for GRPO mode
                eval_metrics, eval_accuracy = eval_on_test_set(
                    model=model,
                    tokenizer=tokenizer, 
                    test_loader=test_loader,
                    eval_class=eval_class,
                    device=device,
                    args=args,
                    round_num=round_num
                )
            else:
                # Evaluation for injection mode - same as standard but with injection generation
                eval_metrics, eval_accuracy = eval_on_test_set_with_injection(
                    model=model,
                    injector=injector,
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

        # Get next question
        question, answer = next(train_loader)

        # Compute loss using appropriate method
        if args.use_hidden_injection:
            total_loss, train_metrics = grpo_loss_with_injection(
                model, base_model, injector, base_injector, tokenizer, 
                question, answer, eval_class, device, round_num, train_log_dir, args
            )
        else:
            total_loss, train_metrics = grpo_loss(
                model, base_model, tokenizer, question, answer, eval_class, 
                device, round_num, train_log_dir, args
            )
        
        # Gradient accumulation and optimization
        total_loss = total_loss / args.gradient_accumulation_steps
        

        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if True: #(round_num + 1) % args.gradient_accumulation_steps == 0:
            if args.use_hidden_injection:
                # Sanity check - get weights before
                weight_before = injector.transform_network[0].weight.data.clone()
                
                # Check if parameters require gradients and have gradients
                
                
                # Check if gradients exist
                # if args.use_hidden_injection:
                #     # Use larger gradient clipping for injection mode
                #     grad_norm = torch.nn.utils.clip_grad_norm_(injector.parameters(), 1.0)  # Increased from 0.1
                # else:
                #     grad_norm = torch.nn.utils.clip_grad_norm_(injector.parameters(), args.max_grad_norm)
                # print(f"Gradient norm before step: {grad_norm}")
                
                
                optimizer.step()
                
                # Sanity check - get weights after and compare
                weight_after = injector.transform_network[0].weight.data
                weight_diff = torch.norm(weight_after - weight_before).item()
                weights_exactly_same = torch.equal(weight_after, weight_before)
                print(f"Weight change norm: {weight_diff}, Exactly same: {weights_exactly_same}")
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()    


        
        # Enhanced logging
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        
        if args.use_hidden_injection:
            grad_norm = torch.nn.utils.clip_grad_norm_(injector.parameters(), float('inf')).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        
        # Log progress periodically
        if round_num % 10 == 0:
            advantage_mean = train_metrics.get('advantage_mean', 0)
            print(f"Step {round_num}: Loss={train_metrics['loss']:.4f}, "
                  f"Advantage={advantage_mean:.4f}, "
                  f"LR={train_metrics['learning_rate']:.2e}")
        
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
        
        # # Save checkpoint periodically
        # if (round_num + 1) % args.save_steps == 0:
        #     if args.use_hidden_injection:
        #         checkpoint_path = os.path.join(args.output_dir, f'injector_checkpoint_{round_num}.pt')
        #         torch.save({
        #             'injector_state_dict': injector.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'round_num': round_num,
        #             'args': args
        #         }, checkpoint_path)
        #         print(f"💾 Saved injector checkpoint to {checkpoint_path}")
        #     else:
        #         checkpoint_path = os.path.join(args.output_dir, f'model_checkpoint_{round_num}.pt')
        #         torch.save({
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'round_num': round_num,
        #             'args': args
        #         }, checkpoint_path)
        #         print(f"💾 Saved model checkpoint to {checkpoint_path}")

    print("Training complete!")
    
    # Save final model/injector
    if args.use_hidden_injection:
        final_path = os.path.join(args.output_dir, 'final_injector.pt')
        torch.save(injector.state_dict(), final_path)
        print(f"💾 Saved final injector to {final_path}")
    else:
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_path)
        print(f"💾 Saved final model to {final_path}")
   
