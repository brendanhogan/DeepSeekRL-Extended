"""
Script to evaluate trained models using a different judge model (GPT-4) to avoid reward hacking.
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface

import llms
import utils
import evaluator
import rldatasets

def eval_with_separate_judge(
    all_models: dict,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    output_dir: str
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set using a different judge model (GPT-4).
    """
    print("Running evaluation with separate judge model...")
    
    total_scores = defaultdict(float)
    num_examples = 0
    total_wins = 0

    # Create new_judge directory for results
    new_judge_dir = os.path.join(output_dir, 'new_judge')
    os.makedirs(new_judge_dir, exist_ok=True)
    
    log_file = os.path.join(new_judge_dir, 'eval_metrics.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        total_debates = 0
        total_wins = 0
        
        for question in tqdm(test_loader, desc="Evaluating with separate judge"):
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
            f.write(f"{prompt_text}\n\n")

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

            # Score completions to get reward metrics
            rewards_per_func, reward_metrics = eval_class.compute_rewards(
                input_prompt=question, 
                all_models=all_models, 
                train_model_completions=completions_text, 
                compare_model_completions=compare_completions_text,
                device=device,
                is_test=True
            )

            # Track total debates and wins
            debates_this_question = len(completions_text)
            total_debates += debates_this_question
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

                # Log if trained model won this debate
                trained_model_won = rewards_per_func[i,0] > 0
                f.write(f"\nOUTCOME: Trained model {'won' if trained_model_won else 'lost'} this debate\n")
                f.write("-"*40 + "\n")

            # Log summary metrics for this question
            f.write("\nSUMMARY METRICS:\n")
            f.write(f"Win rate: {reward_metrics['win_rate']:.2%}\n")
            f.write(f"Number of wins: {reward_metrics['num_wins']}\n")
            f.write(f"Total debates: {reward_metrics['num_debates']}\n")
            f.write(f"Average format scores:\n")
            f.write(f"  Strict format: {reward_metrics['rewards/strict_format']:.4f}\n")
            f.write(f"  Soft format: {reward_metrics['rewards/soft_format']:.4f}\n")
            f.write(f"  XML count: {reward_metrics['rewards/xml_count']:.4f}\n")

            # Update total scores
            for k, v in reward_metrics.items():
                if k.startswith('rewards/'):
                    total_scores[k] += v
        
        # Calculate final metrics
        win_rate = (total_wins / total_debates) * 100 if total_debates > 0 else 0
        avg_scores = {k: v/num_examples for k,v in total_scores.items()}

        # Save metrics
        metrics = {
            'win_rate': win_rate,
            'total_wins': total_wins,
            'total_debates': total_debates,
            'num_examples': num_examples,
            'average_scores': avg_scores
        }

        # Write summary results to file and print
        f.write("\nFINAL EVALUATION RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Total Wins: {total_wins}\n") 
        f.write(f"Total Debates: {total_debates}\n")
        f.write("\nAverage Scores:\n")
        for metric, value in avg_scores.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
        f.write("-" * 20 + "\n")

        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Wins: {total_wins}")
        print(f"Total Debates: {total_debates}")
        print("\nAverage Scores:")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    # Save metrics to JSON
    metrics_path = os.path.join(new_judge_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, win_rate

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    prompt_text: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
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

def parse_args():
    parser = argparse.ArgumentParser(description="Separate judge evaluation arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o", help="Name of model to use as judge")
    parser.add_argument("--compare_model_name", type=str, default="gpt-4o-mini", help="Name of model to use for comparison")
    parser.add_argument("--dataset_name", type=str, default="debate", choices=["debate"], help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="debate", choices=["debate"], help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing trained model checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Get all args 
    args = parse_args() 
    
    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    # Load models
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)
    judge_model = llms.get_judge_model(args.judge_model_name, device)
    compare_model = llms.get_compare_model(args.compare_model_name, device)
    
    # Get test dataset and evaluator
    _, test_loader = rldatasets.get_dataloaders(args.dataset_name)
    eval_class = evaluator.get_evaluator(args.evaluator)

    # First evaluate the untrained model
    print("\nEvaluating untrained model...")
    untrained_models = {
        "training_model": base_model,  # Use base model as untrained model
        "training_model_tokenizer": tokenizer,
        "base_model": base_model,
        "base_model_tokenizer": tokenizer,
        "judge_model": judge_model,
        "compare_model": compare_model
    }
    
    untrained_metrics, untrained_accuracy = eval_with_separate_judge(
        all_models=untrained_models,
        test_loader=test_loader,
        eval_class=eval_class,
        device=device,
        args=args,
        output_dir=os.path.join(args.output_dir, 'untrained')
    )

    # Load the latest checkpoint
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    checkpoints = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(checkpoint_dir) if f.startswith('step_')])
    if not checkpoints:
        raise ValueError("No checkpoints found in output directory")
    
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, f'step_{latest_checkpoint}.pt')
    print(f"\nLoading checkpoint from step {latest_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Now evaluate the trained model
    print("\nEvaluating trained model...")
    trained_models = {
        "training_model": model,
        "training_model_tokenizer": tokenizer,
        "base_model": base_model,
        "base_model_tokenizer": tokenizer,
        "judge_model": judge_model,
        "compare_model": compare_model
    }

    trained_metrics, trained_accuracy = eval_with_separate_judge(
        all_models=trained_models,
        test_loader=test_loader,
        eval_class=eval_class,
        device=device,
        args=args,
        output_dir=os.path.join(args.output_dir, 'trained')
    )

    # Save comparison results
    comparison_dir = os.path.join(args.output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison = {
        'untrained': {
            'metrics': untrained_metrics,
            'accuracy': untrained_accuracy
        },
        'trained': {
            'metrics': trained_metrics,
            'accuracy': trained_accuracy
        },
        'improvement': {
            'win_rate': trained_accuracy - untrained_accuracy,
            'metrics': {
                k: trained_metrics['average_scores'][k] - untrained_metrics['average_scores'][k]
                for k in trained_metrics['average_scores']
            }
        }
    }
    
    with open(os.path.join(comparison_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\nComparison Results:")
    print("-" * 20)
    print(f"Win Rate Improvement: {comparison['improvement']['win_rate']:.2f}%")
    print("\nMetric Improvements:")
    for metric, improvement in comparison['improvement']['metrics'].items():
        print(f"{metric:15s}: {improvement:+.4f}")
    print("-" * 20) 