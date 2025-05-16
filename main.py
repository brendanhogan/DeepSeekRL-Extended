"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
import soundfile as sf
from typing import Optional, Any, Dict
from shutil import copyfile
from collections import defaultdict
from qwen_vl_utils import process_vision_info

from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
# Import necessary modules and constants for PDF generation in main
from PIL import Image as PILImage 
from utils import MAX_COMPLETIONS_PER_PAGE_PDF
from reportlab.platypus import PageBreak # Import PageBreak

import llms
import utils
import evaluator
import rldatasets
from gui_generator import GUIGenerator # For PDF plotting

import re
import math

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
    Evaluate model performance on test set with improved logging and PDF reports.
    (Orchestrates evaluation using helper functions).
    Returns tuple: (aggregated_avg_scores_dict, main_error_metric_value)
    """
    print("Running evaluation on test set...")
    
    # Initialize accumulators for overall, normal, and hard subsets
    num_examples = 0
    num_chains_processed_overall = 0
    num_chains_processed_normal = 0
    num_chains_processed_hard = 0
    aggregated_metrics_sum_overall = defaultdict(float)
    aggregated_metrics_sum_normal = defaultdict(float)
    aggregated_metrics_sum_hard = defaultdict(float)
    
    # For tool usage logging
    all_completions = []
    all_click_success = []
    
    # Ensure temp_vis directory exists for saving images with clicks for PDF
    eval_temp_vis_dir = os.path.join(args.output_dir, 'eval_logs', 'temp_vis')
    os.makedirs(eval_temp_vis_dir, exist_ok=True)
    
    _, pdf_dir, json_dir = utils._setup_eval_directories(args.output_dir)
    # Sanitize experiment name for PDF filename (if present in args.output_dir)
    exp_name_sanitized = os.path.basename(os.path.normpath(args.output_dir)).replace(" ", "_")
    pdf_filename = f'eval_results_round_{round_num}_exp_{exp_name_sanitized}.pdf'
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    doc, styles, story = utils._setup_pdf(pdf_path)
    
    test_loader.reset()

    # Determine the correct error metric key based on dataset type
    if args.dataset_type == 'correlation':
        main_error_metric_key_for_return = 'metrics/mean_abs_correlation_error'
    elif args.dataset_type == 'clock':
        main_error_metric_key_for_return = 'metrics/mean_abs_error_seconds'
    elif args.dataset_type == 'gui':
        main_error_metric_key_for_return = 'metrics/mean_distance_to_center_error' # Or click_hit_rate if preferred
    else:
        main_error_metric_key_for_return = 'reward' # Fallback, should be defined

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on test set")):
        is_hard_example = False # Default for non-GUI tasks
        if args.dataset_type == 'gui':
            img_path, target_details_dict = batch
            prompt_to_use = target_details_dict['dynamic_prompt']
            answer_for_eval_and_pdf = target_details_dict
            is_hard_example = target_details_dict.get('is_hard', False) # Get the flag
        else:
            img_path, answer_string = batch
            prompt_to_use = test_loader.prompt # Static prompt for clock/correlation
            answer_for_eval_and_pdf = answer_string
        # Generate completions
        # Ensure generate_completions uses the correct prompt (dynamic for GUI)
        _, _, _, _, completions_text, _ = generate_completions(
            model, tokenizer, img_path, prompt_to_use, device, args, eval=True, target_details=target_details_dict)
        
        # Add example header to PDF (prompt_to_use will have target name for GUI)
        utils._add_example_header_to_pdf(story, styles, img_path, prompt_to_use, 
                                         answer_for_eval_and_pdf, num_examples, args.dataset_type,
                                         is_hard=is_hard_example)
        
        # For GUI tasks, collect click success information
        batch_click_success = []
        
        for completion_idx, completion_text in enumerate(completions_text):
            # Path for saving image with plotted click (unique per completion)
            vis_image_path_for_pdf = os.path.join(eval_temp_vis_dir, f"round{round_num}_ex{batch_idx}_comp{completion_idx}.png")
            
            # Process single completion also handles plotting for GUI task
            metrics_single = utils._process_single_completion_for_eval(
                completion_text=completion_text, 
                eval_class=eval_class, 
                answer_data=answer_for_eval_and_pdf, # This is target_details for GUI
                device=device, 
                story=story, 
                styles=styles, 
                completion_idx=completion_idx,
                dataset_type=args.dataset_type,
                original_image_path=img_path, # Pass original image path for GUI task
                vis_image_path_for_pdf=vis_image_path_for_pdf, # Path to save visualization
                gui_plotter=GUIGenerator.plot_predictions if args.dataset_type == 'gui' else None
            )
            
            # Log click success for tool usage statistics (if GUI task)
            if args.dataset_type == 'gui' and metrics_single:
                click_hit = metrics_single.get('click_hit', 0.0) > 0
                batch_click_success.append(click_hit)
                
                # Collect for overall logging
                all_completions.append(completion_text)
                all_click_success.append(click_hit)
            
            num_chains_processed_overall += 1
            if metrics_single and isinstance(metrics_single, dict):
                # Aggregate overall metrics
                for metric_name, value in metrics_single.items():
                    if isinstance(value, (int, float)):
                        aggregated_metrics_sum_overall[metric_name] += value
                    elif torch.is_tensor(value) and value.numel() == 1:
                        aggregated_metrics_sum_overall[metric_name] += value.item()
                
                # Aggregate normal/hard metrics
                if is_hard_example:
                    num_chains_processed_hard += 1
                    for metric_name, value in metrics_single.items():
                        if isinstance(value, (int, float)):
                            aggregated_metrics_sum_hard[metric_name] += value
                        elif torch.is_tensor(value) and value.numel() == 1:
                            aggregated_metrics_sum_hard[metric_name] += value.item()
                else:
                    num_chains_processed_normal += 1
                    for metric_name, value in metrics_single.items():
                        if isinstance(value, (int, float)):
                            aggregated_metrics_sum_normal[metric_name] += value
                        elif torch.is_tensor(value) and value.numel() == 1:
                            aggregated_metrics_sum_normal[metric_name] += value.item()
        num_examples += 1
    # --- Calculate final averages --- 
    avg_scores_overall = {}
    avg_scores_normal = {}
    avg_scores_hard = {}

    if num_chains_processed_overall > 0:
        for metric_name, total_value in aggregated_metrics_sum_overall.items():
            avg_scores_overall[f"avg_overall_{metric_name}"] = total_value / num_chains_processed_overall 
    if num_chains_processed_normal > 0:
        for metric_name, total_value in aggregated_metrics_sum_normal.items():
            avg_scores_normal[f"avg_normal_{metric_name}"] = total_value / num_chains_processed_normal
    if num_chains_processed_hard > 0:
         for metric_name, total_value in aggregated_metrics_sum_hard.items():
            avg_scores_hard[f"avg_hard_{metric_name}"] = total_value / num_chains_processed_hard
    
    # Combine all averages into one dictionary for logging
    all_avg_scores = {**avg_scores_overall, **avg_scores_normal, **avg_scores_hard}

    # Get the overall average value of the primary error metric for return
    final_avg_main_metric_value = avg_scores_overall.get(f"avg_overall_{main_error_metric_key_for_return}", float('nan'))

    doc.build(story)
    print(f"PDF report saved to {pdf_path}")

    utils._calculate_and_log_final_metrics(all_avg_scores, json_dir, round_num, args.verbose)
    
    # Log tool usage if this is a GUI task
    if args.dataset_type == 'gui' and all_completions:
        tool_log_file = os.path.join(json_dir, 'tool_usage.json')
        utils.log_tool_usage(
            completions=all_completions,
            click_success=all_click_success,
            output_file=tool_log_file,
            round_num=round_num,
            is_eval=True
        )

    return all_avg_scores, final_avg_main_metric_value # Return all avg scores and the specific main error metric value

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    image_path: str,
    prompt: str,
    device: str,
    args: argparse.Namespace, 
    eval: bool = False,
    target_details: Optional[Dict[str, Any]] = None  # Added parameter for target details
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        image_path: The input image path to generate completions for
        prompt: The input prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        eval: Whether this is an evaluation run
        target_details: Dictionary with target object details (for GUI tasks)
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """

    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}

            ],
        },
    ]

    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)  
    image_inputs, video_inputs = process_vision_info(conversation)

    # Ensure left padding for tokenizer/processor before tokenizing
    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device).to(model.dtype)

    # Repeat input tensors for batch generation
    if eval:
        num_chains = args.num_chains_eval
    else:
        num_chains = args.num_chains
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(num_chains, *([1] * (value.dim() - 1)))
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value 

    # Original prompt_ids/mask are needed for splitting later
    original_prompt_ids = prompt_inputs["input_ids"]

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )

    # Generate all completions at once
    prompt_completion_ids = model.generate(
        **batched_prompt_inputs,
        generation_config=generation_config

    )

    
    # Extract completion ids
    # Use the original prompt length before repeating
    prompt_length = original_prompt_ids.size(1) 
    prompt_ids = prompt_completion_ids[:, :prompt_length] # These are the batched prompt IDs
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Create attention mask based on original prompt mask repeated and completion mask
    prompt_mask = batched_prompt_inputs["attention_mask"] # Use the repeated mask
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Process tool calls if this is a GUI task with target details
    if args.dataset_type == 'gui' and target_details is not None:
        # Define patterns for tool processing
        click_tool_pattern = re.compile(r"tool_name:\s*click_tool\s*\nx:\s*(\d+)\s*\ny:\s*(\d+)", re.MULTILINE)
        check_click_pattern = re.compile(r"tool_name:\s*check_click(?!\s*\ntool_response)", re.MULTILINE)
        
        # Process each completion
        processed_completions = []
        for comp_text in completions_text:
            # Extract the reasoning section
            reasoning_section = ""
            try:
                reasoning_section_parts = comp_text.split("<reasoning>")
                if len(reasoning_section_parts) > 1:
                    reasoning_section = reasoning_section_parts[-1].split("</reasoning>")[0]
            except (IndexError, ValueError):
                # If can't extract reasoning, use the whole text as fallback
                reasoning_section = comp_text
            
            # Check if there's a check_click tool call in the reasoning that doesn't already have a response
            modified_content = comp_text
            check_matches = list(check_click_pattern.finditer(reasoning_section))
            
            if check_matches:
                # Process each check_click call that needs a response
                for check_match in check_matches:
                    # Find the most recent click_tool call before this check_click
                    click_matches = list(click_tool_pattern.finditer(reasoning_section[:check_match.start()]))
                    
                    if click_matches:
                        # Get the most recent click coordinates
                        last_click = click_matches[-1]
                        click_x = int(last_click.group(1))
                        click_y = int(last_click.group(2))
                        
                        # Check if click was successful
                        target_bbox = target_details.get('bounding_box', None)
                        target_name = target_details.get('name', '')
                        
                        if target_bbox:
                            # Check if it's a circular element
                            circular_elements = {"window_close_button", "window_minimize_button", "window_maximize_button"}
                            if target_name in circular_elements:
                                # Use circular hit detection
                                x_min, y_min, x_max, y_max = target_bbox
                                center_x = (x_min + x_max) / 2
                                center_y = (y_min + y_max) / 2
                                radius = (x_max - x_min) / 2
                                distance = math.sqrt((click_x - center_x)**2 + (click_y - center_y)**2)
                                is_hit = distance <= radius
                            else:
                                # Use bounding box hit detection
                                x_min, y_min, x_max, y_max = target_bbox
                                is_hit = (x_min <= click_x <= x_max) and (y_min <= click_y <= y_max)
                        else:
                            is_hit = False
                        
                        # Add tool response line after the check_click tool call
                        response_text = "True" if is_hit else "False"
                        check_end = check_match.end()
                        replacement = f"{check_match.group(0)}\ntool_response: {response_text}"
                        
                        # Replace in the full text
                        before_match = modified_content.split(check_match.group(0))[0]
                        after_match = modified_content.split(check_match.group(0))[1]
                        modified_content = before_match + replacement + after_match
                    else:
                        # No click coordinates found, assume failed check
                        replacement = f"{check_match.group(0)}\ntool_response: False"
                        before_match = modified_content.split(check_match.group(0))[0]
                        after_match = modified_content.split(check_match.group(0))[1]
                        modified_content = before_match + replacement + after_match
            
            processed_completions.append(modified_content)
        
        # Replace original completions with processed ones
        completions_text = processed_completions
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt
    
def score_completions(
    completions_text: list[str],
    prompt_data: Any, # For GUI, this will be target_details_dict including dynamic_prompt
                      # For Clock/Corr, this will be the static prompt string
    image_path: str,
    answer_data: Any, # For GUI, this is target_details_dict (name, bbox, etc.)
                       # For Clock/Corr, this is the answer string (time/R value)
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    """
    log_data = {
        'prompt': {},
        'generations': []
    }

    if args.dataset_type == 'gui':
        # prompt_data is target_details_dict from GUIDataLoader
        # answer_data is also target_details_dict
        log_data['prompt'] = {
            'text': prompt_data['dynamic_prompt'], # The actual prompt shown to LLM
            'target_object_name': answer_data['name'],
            'target_object_bbox': answer_data['bounding_box'],
            'image_path': image_path
        }
        # The `answers` expected by GUIEvaluator.compute_rewards is a list of these answer_data dicts
        eval_answers_list = [answer_data] * len(completions_text)
    else:
        # prompt_data is the static prompt string
        # answer_data is the single answer string (time or R value)
        log_data['prompt'] = {
            'text': prompt_data, 
            'answer': answer_data, 
            'image_path': image_path
        }
        eval_answers_list = [answer_data] * len(completions_text)

    # Format inputs as expected by evaluator
    mock_completions_for_eval = [[{'content': completion}] for completion in completions_text]
    
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=None, # Eval class might not need full prompt structure, depends on its impl.
        completions=mock_completions_for_eval, 
        answers=eval_answers_list, # Pass list of answers (target_details for GUI, string for others)
        device=device
    )

    rewards = rewards_per_func.sum(dim=1)

    for i, (completion, reward_scores_single_chain) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores_single_chain), # Pass scores for one chain
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item() if args.num_chains > 1 else 0.0

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
    args: argparse.Namespace,
    img_path: str,
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str
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
        ref_per_token_logps = utils.get_per_token_logps_vl(base_model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt)

    # Get training model logits
    per_token_logps = utils.get_per_token_logps_vl(model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt)

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
        # For GUI, prompt_info_or_static_prompt will be target_details_dict
        # For Clock/Corr, it will be the static prompt string from the loader
        prompt_info_or_static_prompt: Any, 
        img_path: str,
        # For GUI, answer_details_or_string will be target_details_dict
        # For Clock/Corr, it will be the answer string
        answer_details_or_string: Any, 
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], list[str], torch.Tensor, torch.Tensor]: 
    """
    Compute GRPO loss for a batch.
    Returns: loss, metrics, completions_text, rewards_per_func, advantages
    """
    if args.dataset_type == 'gui':
        # prompt_info_or_static_prompt is target_details_dict
        # answer_details_or_string is also target_details_dict
        current_prompt_text = prompt_info_or_static_prompt['dynamic_prompt']
        current_answer_data = answer_details_or_string # This is the dict for GUIEvaluator
        target_details = prompt_info_or_static_prompt  # Use the complete target details for tool processing
    else:
        current_prompt_text = prompt_info_or_static_prompt
        current_answer_data = answer_details_or_string # This is the string for Clock/Corr
        target_details = None  # No target details for non-GUI tasks

    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = generate_completions(
        model, tokenizer, img_path, current_prompt_text, device, args, target_details=target_details
    )

    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, 
        prompt_info_or_static_prompt, # Pass the original prompt info (dict for GUI, str for others)
        img_path, 
        current_answer_data, # Pass target_details for GUI, answer_str for others
        eval_class, 
        device, 
        args
    )

    log_file_name = f'round_{round_num}_batch_generations.txt' # More descriptive name
    log_file = os.path.join(training_log_dir, log_file_name)
    utils.write_generation_log(log_data, log_file)
    
    # For GUI tasks, log tool usage statistics
    if args.dataset_type == 'gui':
        # Extract click success information from rewards
        click_success = []
        for i, reward_scores in enumerate(rewards_per_func):
            # Use the click_hit reward component (index 1) to determine success
            click_hit = reward_scores[1].item() > 0
            click_success.append(click_hit)
        
        # Log tool usage
        tool_log_file = os.path.join(training_log_dir, 'tool_usage.json')
        utils.log_tool_usage(
            completions=completions_text,
            click_success=click_success,
            output_file=tool_log_file,
            round_num=round_num,
            is_eval=False
        )
    
    image_log_dir = os.path.join(training_log_dir, 'images')
    os.makedirs(image_log_dir, exist_ok=True)
    image_log_path = os.path.join(image_log_dir, f'image_round_{round_num}.png') # One image per round for now
    if not os.path.exists(image_log_path): # Save image only if it hasn't been saved for this round yet
        copyfile(img_path, image_log_path)

    completion_mask = attention_mask[:, prompt_ids.size(1):]
    # Ensure current_prompt_text is used for get_per_token_logps_vl if it relies on it
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args, img_path, tokenizer, current_prompt_text
    )
    metrics.update(loss_metrics)
    
    # Return the required values
    return loss, metrics, completions_text, rewards_per_func, advantages


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model identifier for main and base model")
    parser.add_argument("--dataset_type", type=str, default="clock", choices=["clock", "correlation", "gui"], help="Type of dataset to use ('clock', 'correlation', or 'gui')")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=3000, help="Save a resumable checkpoint every N steps")
    parser.add_argument("--eval_iterations", type=int, default=50, help="Number of iterations for evaluation")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from.")

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
    parser.add_argument("--num_chains_eval", type=int, default=2, help="Number of parallel generation chains for evaluation")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=3000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() 

    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    model, tokenizer = llms.get_llm_tokenizer(args.model_name_or_path, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name_or_path, device)

    print(f"Loading dataset: {args.dataset_type}")
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_type)

    print(f"Loading evaluator for: {args.dataset_type}")
    eval_class = evaluator.get_evaluator(args.dataset_type)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, 'args.json')
    if not args.resume_from_checkpoint or not os.path.exists(args_path):
        args_dict = vars(args)
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
            
    eval_log_dir, eval_pdf_dir, eval_json_dir = utils._setup_eval_directories(args.output_dir)
    train_log_dir = utils._setup_training_log_directory(args.output_dir)
    # Setup dirs for training PDF logs
    train_pdf_dir = os.path.join(train_log_dir, 'pdfs')
    train_temp_vis_dir = os.path.join(train_log_dir, 'temp_vis')
    os.makedirs(train_pdf_dir, exist_ok=True)
    os.makedirs(train_temp_vis_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)
    
    start_round = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'base_model_state_dict' in checkpoint:
                 base_model.load_state_dict(checkpoint['base_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_round = checkpoint['round_num'] + 1
            print(f"Loaded checkpoint. Resuming from round {start_round}")
            temp_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step + start_round))
            scheduler = temp_scheduler 
        else:
            print(f"Warning: Checkpoint file not found at {args.resume_from_checkpoint}. Starting training from scratch.")

    accumulated_loss_val = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    if os.path.exists(os.path.join(train_log_dir, "train_logs.json")):
         with open(os.path.join(train_log_dir, "train_logs.json"), "r") as f:
            try:
                train_metrics_total = json.load(f)
                train_metrics_total = {int(k): v for k, v in train_metrics_total.items()}
            except json.JSONDecodeError:
                 print("Warning: Could not parse existing train_logs.json. Starting with empty logs.")
                 train_metrics_total = {}

    # Variables to store data for the round to be logged in PDF
    pdf_log_round_data = {}

    for round_num in tqdm(range(start_round, args.num_train_iters), initial=start_round, total=args.num_train_iters, desc="Training Progress"):
        
        # # --- Evaluation Step --- (Run periodically)
        if round_num % args.eval_iterations == 0:
            eval_avg_scores, eval_main_metric_val = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            metrics_path = os.path.join(eval_json_dir, f'metrics_round_{round_num}.json') # Use eval_json_dir
            with open(metrics_path, 'w') as f:
                json.dump({
                    'avg_scores': eval_avg_scores,
                    'main_error_metric_value': eval_main_metric_val
                }, f, indent=4)

        # --- Reference Model Update --- (Run periodically)
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # --- Training Step --- (Run every round)
        batch = next(train_loader)
        img_path, data_for_grpo = batch 

        # Perform GRPO step and get data needed for potential PDF logging
        loss, train_metrics, completions_text, rewards_per_func, advantages = grpo_loss(
            model, base_model, tokenizer, 
            data_for_grpo, img_path, data_for_grpo, 
            eval_class, device, round_num, train_log_dir, args
        )
        
        # Store data for potential PDF logging this round
        # This replaces previous round's data, so only the latest is kept
        pdf_log_round_data = {
            'round_num': round_num,
            'img_path': img_path,
            'prompt_info': data_for_grpo, # Contains target_details for GUI, prompt_str otherwise
            'completions_text': completions_text,
            'rewards_per_func': rewards_per_func,
            'advantages': advantages
        }

        # Gradient accumulation and optimizer step
        loss = loss / args.gradient_accumulation_steps 
        loss.backward()
        accumulated_loss_val += loss.item()
        scheduler.step() 
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # --- Checkpoint Saving & Training PDF Logging --- (Run periodically)
        if (round_num + 1) % args.save_steps == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_round_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'base_model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args 
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # --- Generate Training PDF Log for this round --- 
        print(f"Generating training PDF log for round {round_num}...")
        # Use the data stored from the *last* training step of this logging interval
        log_data = pdf_log_round_data
        if log_data.get('round_num') == round_num: # Ensure data is from the correct round
            pdf_train_filename = f'training_log_round_{round_num}.pdf'
            pdf_train_path = os.path.join(train_pdf_dir, pdf_train_filename)
            doc_train, styles_train, story_train = utils._setup_pdf(pdf_train_path)

            # Determine prompt text and answer/target data based on type
            if args.dataset_type == 'gui':
                prompt_text_for_pdf = log_data['prompt_info']['dynamic_prompt']
                answer_data_for_pdf = log_data['prompt_info'] # The dict itself
            else:
                prompt_text_for_pdf = log_data['prompt_info'] # The static prompt string
                answer_data_for_pdf = log_data['prompt_info'] # The answer string (same as prompt_info here)

            # Add PDF Header (Image, Prompt, Target Info)
            utils._add_example_header_to_pdf(
                story_train, styles_train, log_data['img_path'], 
                prompt_text_for_pdf, answer_data_for_pdf, 
                round_num, args.dataset_type
            )

            # Add Completions
            num_chains_logged = min(len(log_data['completions_text']), 8) # Log max 8 chains to keep PDF size manageable
            for compl_idx in range(num_chains_logged):
                completion_text = log_data['completions_text'][compl_idx]
                reward_scores = log_data['rewards_per_func'][compl_idx]
                advantage = log_data['advantages'][compl_idx].item() # Get scalar value
                reward_breakdown = eval_class.get_reward_breakdown(reward_scores) # Get dict

                vis_train_img_path = os.path.join(
                    train_temp_vis_dir, 
                    f"train_round{round_num}_comp{compl_idx}.png"
                )
                
                img_path_for_pdf_entry = None
                # Plot click for GUI task
                if args.dataset_type == 'gui':
                    try:
                        if isinstance(eval_class, evaluator.GUIEvaluator):
                            parsed_click = eval_class._extract_coordinates(completion_text)
                            if parsed_click:
                                pil_img = PILImage.open(log_data['img_path'])
                                plot_data = [{
                                    "name": "VLM Click (Tool Call)", 
                                    "center_x": parsed_click[0], 
                                    "center_y": parsed_click[1],
                                    "is_truth": False
                                }]
                                img_w_click = GUIGenerator.plot_predictions(pil_img, plot_data, pred_color="red")
                                img_w_click.save(vis_train_img_path)
                                img_path_for_pdf_entry = vis_train_img_path
                            else:
                                img_path_for_pdf_entry = log_data['img_path'] # Use original if click not parsed
                        else: 
                            img_path_for_pdf_entry = log_data['img_path']
                    except Exception as plot_err:
                        print(f"  Warning: Error plotting training click for PDF (Round {round_num}, Comp {compl_idx}): {plot_err}")
                        img_path_for_pdf_entry = log_data['img_path']
                else:
                    img_path_for_pdf_entry = log_data['img_path'] # Use original image for non-GUI tasks

                # Add completion details to story
                utils._add_training_completion_to_pdf(
                    story=story_train,
                    styles=styles_train,
                    completion_text=completion_text,
                    reward_breakdown=reward_breakdown,
                    advantage=advantage,
                    completion_idx=compl_idx,
                    image_path_for_completion_pdf=img_path_for_pdf_entry
                )
                # Add PageBreak if needed (e.g., after every 2 completions)
                if (compl_idx + 1) % MAX_COMPLETIONS_PER_PAGE_PDF == 0:
                        story_train.append(PageBreak())
            
            # Build the PDF
            doc_train.build(story_train)
            print(f"Training PDF log saved to {pdf_train_path}")
        else:
                print(f"Warning: Skipping training PDF log for round {round_num} as stored data mismatch.")


        # --- Logging Training Metrics --- (Run every round)
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = loss.item() * args.gradient_accumulation_steps # Log un-normalized loss for the step
        # Calculate grad norm *before* zeroing gradients
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        # Calculate grad norm *after* clipping but *before* optimizer step might be more informative if clipping happens
        # Or calculate before clipping to see raw norm. Let's keep it simple for now:
        # Calculate after backward, before optimizer step/zeroing
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        # Clear cache periodically
        if round_num % 50 == 0:
            torch.cuda.empty_cache()

        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
