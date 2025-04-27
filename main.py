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
from model_interface import ModelInterface
from typing import Optional, List, Dict, Any

import llms
import utils
import evaluator
import rldatasets
import plotter

# Added for detailed log saving
import datetime 

def eval_on_test_set(
    all_models: dict,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float, Optional[List[Dict[str, Any]]]]:
    """
    Evaluate model performance on test set by comparing each model completion
    against a base model completion and having them judged.
    Returns metrics, win rate, and detailed logs.
    """
    print("Running evaluation on test set...")
    
    total_scores = defaultdict(float)
    num_examples = 0
    total_wins = 0
    all_detailed_logs = [] # Store logs from all examples

    # File for human-readable logs
    log_file = os.path.join(args.output_dir, f'eval_log_readable_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        total_comparisons = 0
        total_wins = 0
        
        for example_idx, question in enumerate(tqdm(test_loader, desc="Evaluating on test set")):
            num_examples += 1

            # 1. Prepare prompting
            prompt = [
                {'role': 'system', 'content': test_loader.pre_prompt},
                {'role': 'user', 'content': question}
            ]
            prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)

            # Log Initial prompt 
            f.write("\n" + "="*80 + "\n")
            f.write(f"Example #{num_examples}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Prompt:\n")
            f.write(f"{question}\n\n") # Log the original question, not the templated one

            # Generate completions from trained model
            _, _, _, _, completions_text, _ = generate_completions(
                all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
            )

            # Generate completions for compare model using the interface
            compare_completions_text = []
            for _ in range(args.num_chains):
                completion = all_models["compare_model"].generate(
                    system_prompt=test_loader.pre_prompt,
                    user_prompt=question,
                    max_new_tokens=args.max_completion_length,
                    temperature=args.temperature
                )
                compare_completions_text.append(completion)

            # Score completions to get reward metrics and detailed logs
            rewards_per_func, reward_metrics, detailed_logs_this_q = eval_class.compute_rewards(
                input_prompt=question, 
                all_models=all_models, 
                train_model_completions=completions_text, 
                compare_model_completions=compare_completions_text,
                num_principles=args.num_principles,             # Pass GRM args
                num_inference_rounds=args.number_of_inference_rounds, # Pass GRM args
                device=device,
                is_test=True
            )

            # Store detailed logs if available
            if detailed_logs_this_q:
                # Add example index to each log entry for context
                for log_entry in detailed_logs_this_q:
                    log_entry["example_index"] = example_idx
                all_detailed_logs.extend(detailed_logs_this_q)

            # Track total comparisons and wins
            comparisons_this_question = len(completions_text)
            total_comparisons += comparisons_this_question
            total_wins += reward_metrics['num_wins']

            # For each completion pair, log the results
            for i, (completion, compare_completion) in enumerate(zip(completions_text, compare_completions_text)):
                f.write(f"\nCompletion #{i+1}:\n")
                f.write("-"*40 + "\n\n")

                # Log trained model's response
                f.write("TRAINED MODEL RESPONSE:\n")
                f.write(f"Full response:\n{completion}\n\n")
                
                try:
                    trained_reasoning = completion.split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    trained_answer = completion.split("<answer>\n")[1].split("\n</answer>")[0]
                except:
                    trained_reasoning = "ERROR: Could not parse reasoning"
                    trained_answer = "ERROR: Could not parse answer"
                
                f.write(f"Parsed reasoning:\n{trained_reasoning}\n")
                f.write(f"Parsed answer:\n{trained_answer}\n\n")

                # Log compare model's response
                f.write("COMPARE MODEL RESPONSE:\n")
                f.write(f"Full response:\n{compare_completion}\n\n")
                
                try:
                    compare_reasoning = compare_completion.split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    compare_answer = compare_completion.split("<answer>\n")[1].split("\n</answer>")[0]
                except:
                    compare_reasoning = "ERROR: Could not parse reasoning"
                    compare_answer = "ERROR: Could not parse answer"
                
                f.write(f"Parsed reasoning:\n{compare_reasoning}\n")
                f.write(f"Parsed answer:\n{compare_answer}\n\n")

                # Log reward scores for this completion
                f.write("REWARD SCORES:\n")
                reward_breakdown = eval_class.get_reward_breakdown(rewards_per_func[i])
                for reward_name, reward_value in reward_breakdown.items():
                    f.write(f"{reward_name}: {reward_value:.4f}\n")
                f.write(f"Total reward: {rewards_per_func[i].sum().item():.4f}\n")

                # Log GRM details if available (find the corresponding log entry)
                comparison_log = next((log for log in detailed_logs_this_q if log['comparison_index'] == i), None)
                if comparison_log:
                    f.write("\nGRM EVALUATION DETAILS:\n")
                    f.write(f"Overall Winner: {comparison_log['overall_winner']}\n")
                    for round_detail in comparison_log.get('rounds', []):
                        f.write(f"  Round {round_detail['round']}:\n")
                        if "error" in round_detail:
                            f.write(f"    Error: {round_detail['error']}\n")
                            continue
                        f.write(f"    Principles: {json.dumps(round_detail.get('principles', []))}\n")
                        f.write(f"    Arg1 Critique: {round_detail.get('arg1_critique', 'N/A')} (Score: {round_detail.get('arg1_score', 'N/A')})\n")
                        f.write(f"    Arg2 Critique: {round_detail.get('arg2_critique', 'N/A')} (Score: {round_detail.get('arg2_score', 'N/A')})\n")
                        f.write(f"    Round Winner: {round_detail.get('round_winner', 'N/A')}\n")
                else:
                    # Fallback for older logging or if logs weren't generated
                    trained_model_won = rewards_per_func[i, 0] > 0 # debate_score is index 0
                    f.write(f"\nOUTCOME (Fallback): Trained model {'won' if trained_model_won else 'lost'} this comparison\n")

                f.write("-"*40 + "\n")

            # Log summary metrics for this question
            f.write("\nSUMMARY METRICS:\n")
            f.write(f"Win rate: {reward_metrics['win_rate']:.2%}\n")
            f.write(f"Number of wins: {reward_metrics['num_wins']}\n")
            # f.write(f"Total comparisons: {reward_metrics['num_comparisons']}\n")
            f.write(f"Average format scores:\n")
            f.write(f"  Strict format: {reward_metrics['rewards/strict_format']:.4f}\n")
            f.write(f"  Soft format: {reward_metrics['rewards/soft_format']:.4f}\n")
            f.write(f"  XML count: {reward_metrics['rewards/xml_count']:.4f}\n")

            # Update total scores
            for k, v in reward_metrics.items():
                if k.startswith('rewards/'):
                    total_scores[k] += v

        # Calculate final metrics
        win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
        avg_scores = {k: v/num_examples for k,v in total_scores.items()}

        # Save metrics
        metrics = {
            'win_rate': win_rate,
            'total_wins': total_wins,
            'total_comparisons': total_comparisons,
            'num_examples': num_examples,
            'average_scores': avg_scores
        }

        # Write summary results to file and optionally print
        f.write("\nFINAL EVALUATION RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Total Wins: {total_wins}\n") 
        f.write(f"Total Comparisons: {total_comparisons}\n")
        f.write("\nAverage Scores:\n")
        for metric, value in avg_scores.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
        f.write("-" * 20 + "\n")

        if args.verbose:
            print("\nEvaluation Results:")
            print("-" * 20)
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Wins: {total_wins}")
            print(f"Total Comparisons: {total_comparisons}")
            print("\nAverage Scores:")
            for metric, value in avg_scores.items():
                print(f"{metric:15s}: {value:.4f}")
            print("-" * 20)

    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, win_rate, all_detailed_logs

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    prompt_text: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        prompt_text: The input question/prompt to generate completions for - should be full prompt ready to be turned into token ids (i.e. chat template applied etc)
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

    # Tokenize
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
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics, _ = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=completions_text, 
        compare_model_completions=None,
        num_principles=args.num_principles,
        num_inference_rounds=args.number_of_inference_rounds,
        device=device, 
        is_test=False
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
        train_loader,
        all_models: dict,
        question: str,
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
        loss: The computed GRPO loss per step (before accumulation)
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """

    prompt = [
        {'role': 'system', 'content': test_loader.pre_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)

    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
    )
    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        all_models["training_model"], all_models["base_model"], prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Name/path of base model")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-mini", help="Name of model to use as judge")
    parser.add_argument("--compare_model_name", type=str, default="gpt-4o-mini", help="Name of model to use for comparison")
    parser.add_argument("--dataset_name", type=str, default="debate", choices=["debate", "ld", "chopped"], help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="debate", choices=["debate", "ld", "chopped"], help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=80, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=40, help="Number of iterations for evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

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

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    # GRM Evaluation Parameters (added)
    parser.add_argument("--number_of_inference_rounds", type=int, default=3, help="Number of GRM evaluation rounds per comparison")
    parser.add_argument("--num_principles", type=int, default=3, help="Number of principles to generate per GRM round")

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

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    # Get judge and compare models using the new interfaces
    judge_model = llms.get_judge_model(args.judge_model_name, "cuda:1")
    compare_model = llms.get_compare_model(args.compare_model_name, device)
    
    # Simplified all_models dictionary
    all_models = {
        "training_model": model,
        "training_model_tokenizer": tokenizer,
        "base_model": base_model,
        "base_model_tokenizer": tokenizer,
        "judge_model": judge_model,
        "compare_model": compare_model
    }

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)


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
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        all_models["training_model"].parameters(),
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

    # Resume from checkpoint if requested
    start_round = 0
    if args.resume:
        checkpoints = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(checkpoint_dir) if f.startswith('step_')])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{latest_checkpoint}.pt')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_round = checkpoint['round_num'] + 1
            train_metrics_total = checkpoint['train_metrics_total']
            print(f"Resuming from checkpoint at step {latest_checkpoint}")
        else:
            print("No checkpoints found, starting from scratch")
            train_metrics_total = {}
    else:
        train_metrics_total = {}

    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()

    # Store eval results over time
    evaluation_results = {}

    for round_num in tqdm(range(start_round, args.num_train_iters), desc="Training Progress"):
        if args.verbose: print(f"Starting Round {round_num}")
        
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0:
            if args.verbose: print(f"Running evaluation at round {round_num}...")
            eval_metrics, eval_win_rate, detailed_eval_logs = eval_on_test_set(
                all_models=all_models,
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save detailed logs to a separate JSON file
            detailed_log_path = os.path.join(eval_log_dir, f'detailed_eval_log_{round_num}.json')
            with open(detailed_log_path, 'w') as f_log:
                json.dump(detailed_eval_logs, f_log, indent=2)
            
            # Save summary metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'summary_metrics_{round_num}.json')
            pdf_output_path = os.path.join(eval_log_dir, f'evaluation_report_{round_num}.pdf') # PDF path
            with open(metrics_path, 'w') as f_metrics:
                json.dump({
                    'round': round_num,
                    'win_rate': eval_win_rate,
                    'metrics': eval_metrics
                }, f_metrics, indent=4)
            evaluation_results[round_num] = {'win_rate': eval_win_rate, 'metrics': eval_metrics}

            # Generate PDF report
            utils.create_evaluation_pdf(
                detailed_log_path=detailed_log_path,
                summary_metrics_path=metrics_path,
                win_rate_plot_path=os.path.join(args.output_dir, 'eval_win_rate.png'), # Pass plot path
                output_pdf_path=pdf_output_path
            )

            if args.verbose: print(f"Evaluation complete for round {round_num}. Win rate: {eval_win_rate:.2f}%")

        # Save checkpoint
        if (round_num + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics_total': train_metrics_total
            }, checkpoint_path)

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get next question
        question = next(train_loader)

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(train_loader, all_models, question, eval_class, device, round_num, train_log_dir, args)
        print(f"Total loss: {total_loss}")

        # Normalize loss by accumulation steps
        loss_for_step = total_loss / args.gradient_accumulation_steps
        loss_for_step.backward()
        accumulated_loss += loss_for_step.item() # Log the normalized loss

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() # Step scheduler here
            if args.verbose: print(f"Optimizer step at round {round_num}. Accumulated Loss: {accumulated_loss:.4f}")
            accumulated_loss = 0 # Reset accumulated loss

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = loss_for_step.item() # Log the loss for this step
        train_metrics["accumulated_loss_before_step"] = accumulated_loss if (round_num + 1) % args.gradient_accumulation_steps != 0 else train_metrics["loss"] # approx
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
        print("1 train step complete")

    # Generate final plots after training finishes
    print("\nGenerating final training plots...")
    try:
        plotter.plot_metrics(args.output_dir)
    except Exception as e:
        print(f"Error generating final plots: {e}")

    # Final evaluation after training completes
    print("\nTraining finished. Running final evaluation...")
    final_eval_metrics, final_eval_win_rate, final_detailed_eval_logs = eval_on_test_set(
        all_models=all_models,
        test_loader=test_loader,
        eval_class=eval_class,
        device=device,
        args=args,
        round_num=args.num_train_iters - 1
    )

    # Generate final PDF report
    if final_detailed_eval_logs: # Only generate if logs exist
         try:
            final_win_rate_plot_path = os.path.join(args.output_dir, 'eval_win_rate.png')
            utils.create_evaluation_pdf(
                detailed_log_path=os.path.join(eval_log_dir, f'detailed_eval_log_{args.num_train_iters - 1}.json'),
                summary_metrics_path=os.path.join(eval_log_dir, f'summary_metrics_{args.num_train_iters - 1}.json'),
                win_rate_plot_path=final_win_rate_plot_path if os.path.exists(final_win_rate_plot_path) else None, # Pass plot path if it exists
                output_pdf_path=os.path.join(eval_log_dir, f'evaluation_report_{args.num_train_iters - 1}.pdf')
            )
         except Exception as e:
             print(f"Error calling PDF generation for final report: {e}")

