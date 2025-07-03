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

def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    inference_interface: llms.InferenceModelInterface,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int, 
    is_first_round,
) -> tuple[dict[str, float], float]:
    """
    Evaluate strategy generation model performance on test set.
    """
    print("Running strategy evaluation on test set...")
    
    total_correct = 0
    num_examples = 0
    all_eval_results = []

    test_loader.reset()
    
    for question, answer in tqdm(test_loader, desc="Strategy evaluation"):
            
        # Generate strategies
        # if is_first_round:
        #     strategies_text = ["think step by step"] * args.num_chains
        # else:
        #     _, _, _, _, strategies_text, _ = generate_strategies(
        #         model, tokenizer, question, device, args
        #     )

        _, _, _, _, strategies_text, _ = generate_strategies(
            model, tokenizer, question, device, args
        )

        # Eval just top 1 
        strategies_text = [strategies_text[0], strategies_text[1]]

        # Evaluate each strategy
        strategy_results = []
        total_correct_for_question = 0

        
        for strategy in tqdm(strategies_text, desc="Evaluating strategies", leave=False):
            success_score, detailed_results = evaluate_strategy_with_inference(
                strategy, question, answer, inference_interface, eval_class
            )
            strategy_results.append(detailed_results)
            total_correct_for_question += success_score
        
        # Use best strategy result for this question
        best_success = max(result['success'] for result in strategy_results)
        total_correct += best_success
        
        eval_result = {
            'question': question,
            'answer': answer,
            'strategies': strategy_results,
            'best_success': best_success
        }
        all_eval_results.append(eval_result)
        num_examples += 1


    # Calculate final metrics
    accuracy = total_correct / num_examples if num_examples > 0 else 0.0

    # Save detailed results
    eval_log_file = os.path.join(args.output_dir, f'strategy_eval_{round_num}.json')
    with open(eval_log_file, 'w') as f:
        json.dump({
            'round': round_num,
            'accuracy': accuracy,
            'num_examples': num_examples,
            'results': all_eval_results
        }, f, indent=2, ensure_ascii=False)

    # Save human-readable summary
    text_log_file = os.path.join(args.output_dir, f'strategy_eval_{round_num}.txt')
    with open(text_log_file, 'w') as f:
        f.write(f"STRATEGY EVALUATION - Round {round_num}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.1%} ({total_correct}/{num_examples})\n\n")
        
        f.write("SAMPLE EVALUATIONS:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(all_eval_results[:3]):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Q: {result['question']}\n")
            f.write(f"A: {result['answer']}\n")
            f.write(f"Best Success: {result['best_success']}\n")
            for j, strategy_result in enumerate(result['strategies'][:2]):
                f.write(f"  Strategy {j+1}: {strategy_result['strategy']}\n")
                f.write(f"  Correct: {strategy_result['correct']}\n")
            f.write("-" * 20 + "\n")

    metrics = {'accuracy': accuracy}
    
    if args.verbose:
        print(f"\nStrategy Evaluation Results:")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.1%}")
        print("-" * 30)

    return metrics, accuracy * 100

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

def generate_strategies(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate strategy advice for solving math problems.
    
    Returns:
        prompt_strategy_ids: Full sequence tokens
        prompt_ids: Just the prompt tokens  
        strategy_ids: Just the strategy tokens
        attention_mask: Attention mask for full sequence
        strategies_text: Decoded strategy texts
        prompt_text: The formatted prompt
    """
    # Strategy generation prompt
    prompt = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': f'You are in charge of giving strategic advice to another Large Language Model on how to answer the problem. The large language model will be given a numerical math question - that has a single integer as its answer. The model will be graded based off how correct the answer is, as well as if it first provided its reasoning in <reasoning></reasoning> tags, and its final answer in <answer></answer> tags - it is important that the answer tags contain just the integer, no other characters or statements. You should provide the model with the advice it needs to answer this question and get full points - respond with only your advice to the model. Here is the question: {question}'}
    ]
    
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate and repeat for multiple generations
    # prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    # prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Generate strategies
    generation_config = GenerationConfig(
        max_new_tokens=args.max_strategy_tokens,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    prompt_strategy_ids = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config
    )

    # Extract strategy tokens
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_strategy_ids[:, :prompt_length]
    strategy_ids = prompt_strategy_ids[:, prompt_length:]

    # Create strategy mask (everything up to EOS)
    is_eos = strategy_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    strategy_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, strategy_mask], dim=1)

    # Decode strategies
    strategies_text = tokenizer.batch_decode(strategy_ids, skip_special_tokens=True)

    return prompt_strategy_ids, prompt_ids, strategy_ids, attention_mask, strategies_text, prompt_text


def evaluate_strategy_with_inference(
    strategy: str,
    question: str,
    answer: str,
    inference_interface: llms.InferenceModelInterface,
    eval_class: evaluator.RewardEvaluator
) -> tuple[float, dict]:
    """
    Evaluate a strategy with the inference model.
    
    Returns:
        success_score: 1.0 if solution is correct, 0.0 otherwise
        detailed_results: Dict with solution and metrics
    """
    # Handle prompting differently for local vs OpenAI models
    # OpenAI: Complex system prompt + simple user prompt  
    # Local: Simple system prompt + complex user prompt (better for chat models)
    if inference_interface.is_openai:
        # OpenAI models: use system prompt as before
        system_prompt = f"""You will be given a math problem. Here is some strategic advice:
<strategy>{strategy}</strategy>

Now solve this problem. Format your response as:
<reason>Your reasoning here</reason>
<answer>Your final answer</answer>

"""
        
        user_prompt = question
    else:
        # Local models: simple system prompt, everything in user prompt
        system_prompt = "You are Qwen, a helpful assistant."
        
        user_prompt = f"""You will be given a math problem. Here is some strategic advice:
<strategy>{strategy}</strategy>

Now solve this problem. Format your response as:
<reason>Your reasoning here</reason>
<answer>Your final answer</answer>



Here is the math problem: {question}"""

    # Generate single solution
    solution = inference_interface.generate(
        user_prompt,
        max_tokens=500,
        system_prompt=system_prompt
    )
    
    # Score this solution
    mock_prompts = [[{'content': question}]]
    mock_completions = [[{'content': solution}]]
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=[answer],
        device="cpu"
    )
    
    is_correct = metrics['accuracy'] == 1.0
    success_score = 1.0 if is_correct else 0.0
    
    detailed_results = {
        'strategy': strategy,
        'question': question,
        'answer': answer,
        'solution': solution,
        'solution_length': len(solution),
        'solution_has_reason_tags': '<reason>' in solution and '</reason>' in solution,
        'solution_has_answer_tags': '<answer>' in solution and '</answer>' in solution,
        'correct': is_correct,
        'total_reward': rewards_per_func.sum().item(),
        'metrics': metrics,
        'success': success_score,
        'prompts': {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'full_conversation': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        },
        'debug_info': {
            'model_type': 'local' if not inference_interface.is_openai else 'openai',
            'raw_solution_preview': solution[:200] + ('...' if len(solution) > 200 else ''),
            'solution_word_count': len(solution.split()),
            'contains_xml_tags': any(tag in solution for tag in ['<reason>', '</reason>', '<answer>', '</answer>'])
        }
    }
    
    return success_score, detailed_results


def score_strategies(
    strategies_text: list[str],
    question: str,
    answer: str,
    inference_interface: llms.InferenceModelInterface,
    eval_class: evaluator.RewardEvaluator,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Score strategies based on inference model performance.
    """
    rewards = []
    log_data = {
        'question': question,
        'answer': answer,
        'strategy_evaluations': []
    }
    
    total_correct = 0
    
    for strategy in strategies_text:
        success_score, detailed_results = evaluate_strategy_with_inference(
            strategy, question, answer, inference_interface, eval_class
        )
        
        rewards.append(success_score * 2.0)  # Scale to match original reward scale
        total_correct += success_score
        log_data['strategy_evaluations'].append(detailed_results)
    
    rewards = torch.tensor(rewards, dtype=torch.float32, device="cpu")
    
    # Compute advantages (same logic as before)
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)
    
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    
    metrics = {
        'success_rate': total_correct / len(strategies_text),
        'mean_reward': rewards.mean().item(),
        'reward_std': std_grouped_rewards.mean().item()
    }
    
    return rewards, advantages, metrics, log_data


def compute_strategy_loss(
    model: PreTrainedModel,
    prompt_strategy_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    strategy_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss only over strategy tokens.
    """
    logits_to_keep = strategy_ids.size(1)


    # Get training model logits
    input_ids = torch.cat([prompt_ids, strategy_ids], dim=1)
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute loss with advantages (only over strategy tokens)
    advantages = advantages.cuda()
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss)
    loss = ((per_token_loss * strategy_mask).sum(dim=1) / strategy_mask.sum(dim=1)).mean()

    # Metrics
    metrics = {}
    strategy_length = strategy_mask.sum(1).float().mean().item()
    metrics["strategy_length"] = strategy_length

    return loss, metrics


def strategy_grpo_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    inference_interface: llms.InferenceModelInterface,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str, 
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss for strategy generation.
    """
    # Generate strategies
    prompt_strategy_ids, prompt_ids, strategy_ids, attention_mask, strategies_text, prompt_text = generate_strategies(
        model, tokenizer, question, device, args
    )

    # Score strategies using inference model
    rewards, advantages, metrics, log_data = score_strategies(
        strategies_text, question, answer, inference_interface, eval_class, args
    )

    # Write detailed log
    log_file = os.path.join(training_log_dir, f'{round_num}_strategy_evaluation.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Write human-readable log
    text_log_file = os.path.join(training_log_dir, f'{round_num}_strategy_evaluation.txt')
    with open(text_log_file, 'w') as f:
        f.write(f"STRATEGY EVALUATION - Round {round_num}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n\n")
        
        for i, eval_result in enumerate(log_data['strategy_evaluations']):
            f.write(f"Strategy {i+1}: {eval_result['strategy']}\n")
            f.write(f"Correct: {eval_result['correct']}\n")
            f.write(f"System Prompt: {eval_result['prompts']['system_prompt'][:100]}...\n")
            f.write(f"FULL MODEL OUTPUT ({len(eval_result['solution'])} chars):\n")
            f.write("=" * 60 + "\n")
            f.write(f"{eval_result['solution']}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Has XML formatting: {eval_result.get('solution_has_reason_tags', False) and eval_result.get('solution_has_answer_tags', False)}\n")
            f.write(f"Word count: {eval_result.get('solution_word_count', 'unknown')}\n")
            f.write(f"Reward Breakdown: {eval_result['metrics']}\n")
            f.write("\n")

    # Compute loss
    strategy_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_strategy_loss(
        model, prompt_strategy_ids, prompt_ids, strategy_ids,
        attention_mask, strategy_mask, advantages, args
    )

    metrics.update(loss_metrics)
    return loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--inference_model", type=str, default="gpt-4.1-nano", help="Name/path of inference model for evaluation")
    parser.add_argument("--use_local_inference_model", action="store_true", help="Use Qwen2.5-7B-Instruct for inference instead of GPT-4.1-nano")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset to use for training (gsm8k or math500)")
    parser.add_argument("--evaluator", type=str, default="gsm8k", help="Evaluator to use for scoring (gsm8k or math500)")



    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=150, help="Save model every N steps")
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
    parser.add_argument("--num_chains", type=int, default=4, help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")
    parser.add_argument("--max_strategy_tokens", type=int, default=200, help="Maximum tokens for strategy generation")

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

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)

    # Set inference model based on flag
    if args.use_local_inference_model:
        print("Using local Qwen2.5-7B-Instruct for inference")
        inference_model_name = "Qwen/Qwen2.5-7B-Instruct"
    else:
        print(f"Using {args.inference_model} for inference")
        inference_model_name = args.inference_model
        
    inference_model_interface = llms.get_inference_model_interface(inference_model_name, device)

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)

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


    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

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
    
        # # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                inference_interface=inference_model_interface,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num, 
                is_first_round = round_num == 0
            )

            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # # Slowly update ref model
        # if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
        #     with torch.no_grad():
        #         for param, ref_param in zip(model.parameters(), base_model.parameters()):
        #             ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get next question
        question, answer = next(train_loader)

        # Do strategy GRPO - generate strategies, score with inference model, compute advantage, compute loss 
        total_loss, train_metrics = strategy_grpo_loss(model, tokenizer, question, answer, inference_model_interface, eval_class, device, round_num, train_log_dir, args)
        
        # Gradient accumulation
        total_loss = total_loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()



        # # Step optimizer
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
       
