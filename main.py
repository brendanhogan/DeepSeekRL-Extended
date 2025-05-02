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
import re
import cairosvg
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportlabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import html
import gc
import shutil
from datetime import datetime, timedelta

import llms
import utils
import evaluator
import rldatasets

def render_svg_or_fallback(completion_text: str, output_png_path: str):
    """
    Extracts SVG from <answer> tags, renders to PNG if valid, or saves a fallback.
    If even fallback generation fails, returns None.

    Args:
        completion_text: The raw text output from the model.
        output_png_path: The desired path to save the output PNG.

    Returns:
        The path to the saved PNG (rendered or fallback), or None on critical failure.
    """
    try: 
        # If anything in here fails, we return an all black image 
        answer_match = re.search(r"<answer>(.*?)</answer>", completion_text, re.DOTALL | re.IGNORECASE)
        extracted_svg = answer_match.group(1).strip()
        cairosvg.svg2png(
            bytestring=extracted_svg.encode('utf-8'),
            write_to=output_png_path,
            output_width=224,
            output_height=224
        )
        return output_png_path
    except:
        black_img = Image.new('RGB', (224, 224), (0, 0, 0))
        black_img.save(output_png_path)
        black_img.close()  # Properly close the image to free resources
        return output_png_path

def eval_on_test_set(
    all_models: dict,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set by comparing each model completion
    against a base model completion and having them judged. Generates a PDF report.
    """
    print("Running evaluation on test set...")
    
    total_scores = defaultdict(float)
    num_examples = 0
    total_comparisons = 0 # Renamed from total_wins in original code, seems it was counting comparisons
    total_wins = 0

    # Setup PDF document
    pdf_path = os.path.join(args.output_dir, "eval_logs", f'eval_report_{round_num}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    # Custom styles
    styles.add(ParagraphStyle(name='CodeStyle', parent=styles['Code'], fontName='Courier', fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='HeaderStyle', parent=styles['h1'], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='SubHeaderStyle', parent=styles['h2'], alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='NormalLeft', parent=styles['Normal'], alignment=TA_LEFT))

    story = []
    story.append(Paragraph(f"Evaluation Report - Round {round_num}", styles['HeaderStyle']))
    story.append(Spacer(1, 0.3*inch))

    output_dir_eval = os.path.join(args.output_dir,"eval_logs", f'eval_images_{round_num}')
    os.makedirs(output_dir_eval, exist_ok=True)

    test_loader.reset()
    
    for question in tqdm(test_loader, desc="Evaluating on test set"):
        num_examples += 1

        # --- PDF Section: Question Header ---
        story.append(Paragraph(f"Question {num_examples}", styles['SubHeaderStyle']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Setting:", styles['h3']))
        story.append(Paragraph(html.escape(question), styles['CodeStyle']))
        story.append(Spacer(1, 0.1*inch))

        # 1. Prepare prompting
        prompt = [
            {'role': 'system', 'content': test_loader.pre_prompt},
            {'role': 'user', 'content': f"Scene Description: {question}"}
        ]
        prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,enable_thinking=False)

        # --- PDF Section: Full Prompt ---
        story.append(Paragraph("Full Prompt (Input to Both Models):", styles['h3']))
        story.append(Paragraph(html.escape(prompt_text), styles['CodeStyle'])) # Use <br/> for newlines in PDF
        story.append(Spacer(1, 0.2*inch))

        ###########################################
        ## Generate completions + render images ##
        ###########################################
        # Generate completions from trained model
        _, _, _, _, completions_text, _ = generate_completions(all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args, args.eval_num_chains)

        train_model_image_paths = []
        for i, completion in enumerate(completions_text):
            img_path = os.path.join(output_dir_eval, f"trained_model_example_{num_examples}_completion_{i}.png")
            train_model_image_paths.append(render_svg_or_fallback(completion, img_path))

        # Generate completions for compare model using the interface
        compare_completions_text = []
        for _ in range(args.eval_num_chains):
            completion = all_models["compare_model"].generate(
                system_prompt=test_loader.pre_prompt,
                user_prompt=question,
                max_new_tokens=args.max_completion_length,
                temperature=args.temperature
            )
            compare_completions_text.append(completion)

        compare_model_image_paths = []
        for i, completion in enumerate(compare_completions_text):
            img_path = os.path.join(output_dir_eval, f"compare_model_example_{num_examples}_completion_{i}.png")
            compare_model_image_paths.append(render_svg_or_fallback(completion, img_path))

        ###########################################
        ## Score completions + get reward metrics ##
        ###########################################
        rewards_per_func, reward_metrics = eval_class.compute_rewards(
            input_prompt=question, 
            all_models=all_models, 
            train_model_completions=completions_text, 
            compare_model_completions=compare_completions_text,
            train_model_image_paths=train_model_image_paths,
            compare_model_image_paths=compare_model_image_paths,
            device=device,
            is_test=True
        )

        # Track total comparisons and wins
        comparisons_this_question = len(completions_text)
        total_comparisons += comparisons_this_question
        total_wins += reward_metrics['num_wins'] # Assuming reward_metrics has 'num_wins'

        # --- PDF Section: Per Completion Details ---
        for i, (completion, compare_completion) in enumerate(zip(completions_text, compare_completions_text)):
            story.append(Paragraph(f"Completion {i+1}", styles['SubHeaderStyle']))
            story.append(Spacer(1, 0.1*inch))

            # Trained model details
            story.append(Paragraph("Trained Model Response:", styles['h3']))
            story.append(Paragraph(html.escape(completion), styles['CodeStyle']))
            try:
                trained_reasoning = completion.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                trained_answer = completion.split("<answer>")[1].split("</answer>")[0].strip()
                story.append(Paragraph("Parsed Reasoning:", styles['h4']))
                story.append(Paragraph(html.escape(trained_reasoning), styles['CodeStyle']))
                story.append(Paragraph("Parsed Answer:", styles['h4']))
                story.append(Paragraph(html.escape(trained_answer), styles['CodeStyle']))
            except Exception:
                story.append(Paragraph("ERROR: Could not parse reasoning/answer.", styles['CodeStyle']))
            story.append(Spacer(1, 0.1*inch))

            # Compare model details
            story.append(Paragraph("Compare Model Response:", styles['h3']))
            story.append(Paragraph(html.escape(compare_completion), styles['CodeStyle']))
            try:
                compare_reasoning = compare_completion.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                compare_answer = compare_completion.split("<answer>")[1].split("</answer>")[0].strip()
                story.append(Paragraph("Parsed Reasoning:", styles['h4']))
                story.append(Paragraph(html.escape(compare_reasoning), styles['CodeStyle']))
                story.append(Paragraph("Parsed Answer:", styles['h4']))
                story.append(Paragraph(html.escape(compare_answer), styles['CodeStyle']))
            except Exception:
                 story.append(Paragraph("ERROR: Could not parse reasoning/answer.", styles['CodeStyle']))
            story.append(Spacer(1, 0.1*inch))

            # Images side-by-side
            try:
                img1 = ReportlabImage(train_model_image_paths[i], width=2*inch, height=2*inch)
                img2 = ReportlabImage(compare_model_image_paths[i], width=2*inch, height=2*inch)
                img_table_data = [
                    [Paragraph("Trained Model Image", styles['NormalLeft']), Paragraph("Compare Model Image", styles['NormalLeft'])],
                    [img1, img2]
                ]
                img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('VALIGN', (0, 1), (-1, -1), 'TOP'),
                    ('BOX', (0, 1), (0, 1), 0.25, colors.black), # Box around first image
                    ('BOX', (1, 1), (1, 1), 0.25, colors.black), # Box around second image
                ]))
                story.append(Paragraph("Judge Input Images:", styles['h3']))
                story.append(img_table)
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                story.append(Paragraph(f"Error displaying images: {e}", styles['CodeStyle']))

            # Judging Result
            trained_model_won = rewards_per_func[i, 0] > 0 # Assuming first element indicates win/loss
            outcome_text = "TRAINED MODEL WINS" if trained_model_won else "TRAINED MODEL LOSES"
            story.append(Paragraph(f"Outcome: {outcome_text}", styles['h3']))
            story.append(Spacer(1, 0.1*inch))

            # Reward Scores Breakdown
            story.append(Paragraph("Reward Scores:", styles['h3']))
            reward_breakdown = eval_class.get_reward_breakdown(rewards_per_func[i])
            for reward_name, reward_value in reward_breakdown.items():
                story.append(Paragraph(f"{reward_name}: {reward_value:.4f}", styles['CodeStyle']))
            story.append(Paragraph(f"Total reward: {rewards_per_func[i].sum().item():.4f}", styles['CodeStyle']))
            story.append(Spacer(1, 0.2*inch)) # Spacer after each completion block

        # Update total scores for summary
        for k, v in reward_metrics.items():
            if k.startswith('rewards/'):
                total_scores[k] += v

    # Calculate final metrics
    win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    avg_scores = {k: v/num_examples for k,v in total_scores.items() if num_examples > 0}

    # Final metrics dictionary (for JSON logging)
    metrics = {
        'win_rate': win_rate,
        'total_wins': total_wins,
        'total_comparisons': total_comparisons,
        'num_examples': num_examples,
        'average_scores': avg_scores
    }

    # --- PDF Section: Final Summary ---
    story.append(Paragraph("FINAL EVALUATION RESULTS", styles['SubHeaderStyle']))
    story.append(Spacer(1, 0.2*inch))
    summary_text = [
        f"Win Rate: {win_rate:.2f}%",
        f"Total Wins: {total_wins}",
        f"Total Comparisons: {total_comparisons}",
        f"Number of Questions: {num_examples}",
        "\nAverage Scores:"
    ]
    for metric, value in avg_scores.items():
       summary_text.append(f"  {metric.replace('rewards/', ''):<15}: {value:.4f}")

    story.append(Paragraph("<br/>".join(html.escape(line) for line in summary_text), styles['CodeStyle']))

    doc.build(story)
    
    # Clean up memory-intensive objects
    del story
    del doc
    torch.cuda.empty_cache()
    
    # Keep JSON logging
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    

    return metrics, win_rate

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    prompt_text: str,
    device: str,
    args: argparse.Namespace, 
    num_chains: int = 1
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
    prompt_ids = prompt_ids.repeat(num_chains, 1)
    prompt_mask = prompt_mask.repeat(num_chains, 1)

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
    args: argparse.Namespace,
    train_model_image_paths: list[str]
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
        pairwise_results: List containing pairwise comparison results (or None)
        wins_list: List containing win counts for each completion (or None)
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
    rewards_per_func, metrics, pairwise_results, wins_list = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=completions_text, 
        train_model_image_paths=train_model_image_paths,
        compare_model_completions=None,
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

    return rewards, advantages, rewards_per_func, metrics, log_data, pairwise_results, wins_list

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
        test_loader,
        train_loader,
        all_models: dict,
        question: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss and log training details to a PDF for the round.
    
    Args:
        test_loader: DataLoader for test set (to get pre_prompt)
        train_loader: DataLoader for train set
        all_models: Dictionary containing all models and tokenizers
        question: Input question/prompt
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs and PDF
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
    """

    # --- PDF Setup ---
    pdf_path = os.path.join(training_log_dir, f'training_report_round_{round_num}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    # Custom styles (same as eval)
    styles.add(ParagraphStyle(name='CodeStyle', parent=styles['Code'], fontName='Courier', fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='HeaderStyle', parent=styles['h1'], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='SubHeaderStyle', parent=styles['h2'], alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='NormalLeft', parent=styles['Normal'], alignment=TA_LEFT))
    # Add a smaller code style specifically for the conversation log
    styles.add(ParagraphStyle(name='SmallCodeStyle', parent=styles['Code'], fontName='Courier', fontSize=6, leading=7))
    story = []
    story.append(Paragraph(f"Training Report - Round {round_num}", styles['HeaderStyle']))
    story.append(Spacer(1, 0.3*inch))

    # List to keep track of opened image files for manual closing
    opened_files = []

    # --- Prompt Formatting & PDF Logging ---
    prompt = [
        {'role': 'system', 'content': test_loader.pre_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    story.append(Paragraph("Full Prompt:", styles["SubHeaderStyle"]))
    story.append(Paragraph(html.escape(prompt_text), styles["CodeStyle"]))
    story.append(Spacer(1, 0.2*inch))

    # --- Generate Completions & Render Images ---
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args, args.num_chains
    )
    
    image_dir = os.path.join(training_log_dir, "images", f"round_{round_num}")
    os.makedirs(image_dir, exist_ok=True)
    
    train_model_image_paths = []
    for i, completion in enumerate(completions_text):
        img_path = os.path.join(image_dir, f"completion_{i}.png")
        render_svg_or_fallback(completion, img_path)
        train_model_image_paths.append(img_path)

    # --- Score Completions ---
    rewards, advantages, rewards_per_func, metrics, log_data, pairwise_results, wins_list = score_completions(
        completions_text, question, eval_class, device, args, train_model_image_paths
    )


    # --- PDF Logging: Individual Completions ---
    if round_num % 50 == 0:
        story.append(Paragraph("Generated Completions & Scores:", styles["SubHeaderStyle"]))
        story.append(Spacer(1, 0.1*inch))
        completion_details = [] # For ranking later

        for i, completion in enumerate(completions_text):
            story.append(Paragraph(f"Completion {i+1}", styles["h3"]))
            
            # Full Response
            story.append(Paragraph("Full Response:", styles["h4"]))
            story.append(Paragraph(html.escape(completion), styles['CodeStyle']))
            story.append(Spacer(1, 0.05*inch))
            
            # Parsed Sections
            try:
                trained_reasoning = completion.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                trained_answer = completion.split("<answer>")[1].split("</answer>")[0].strip()
                story.append(Paragraph("Parsed Reasoning:", styles["h4"]))
                story.append(Paragraph(html.escape(trained_reasoning), styles["CodeStyle"]))
                story.append(Paragraph("Parsed Answer:", styles["h4"]))
                story.append(Paragraph(html.escape(trained_answer), styles["CodeStyle"]))
            except Exception:
                story.append(Paragraph("ERROR: Could not parse reasoning/answer.", styles["CodeStyle"]))
            story.append(Spacer(1, 0.05*inch))
            
            # Image
            img_obj = None
            try:
                img_path = train_model_image_paths[i]
                f = open(img_path, 'rb')
                opened_files.append(f)
                img_obj = ReportlabImage(f, width=1.5*inch, height=1.5*inch)
                story.append(Paragraph("Generated Image:", styles["h4"]))
                story.append(img_obj)
            except Exception as e:
                story.append(Paragraph(f"Error displaying image {i} ({os.path.basename(train_model_image_paths[i])}): {e}", styles["CodeStyle"]))
            story.append(Spacer(1, 0.05*inch))
            
            # Scores
            story.append(Paragraph("Reward Scores:", styles["h4"]))
            reward_breakdown = eval_class.get_reward_breakdown(rewards_per_func[i])
            primary_score = rewards_per_func[i, 0].item() # Assume first score is primary for ranking
            for reward_name, reward_value in reward_breakdown.items():
                story.append(Paragraph(f"{reward_name}: {reward_value:.4f}", styles["CodeStyle"]))
            story.append(Paragraph(f"Total reward: {rewards[i].item():.4f}", styles["CodeStyle"]))
            story.append(Paragraph(f"Advantage: {advantages[i].item():.4f}", styles["CodeStyle"]))
            story.append(Spacer(1, 0.2*inch))
            
            # Collect details for ranking
            completion_details.append({
                'score': primary_score,
                'text': completion,
                'image_path': train_model_image_paths[i],
                'index': i,
                'wins': wins_list[i] if wins_list else 0 # Add win count
            })
            
            # No need to del img_obj here, file handle is managed in opened_files

        # --- PDF Logging: Ranked List ---
        story.append(Paragraph("Ranked Completions (by Primary Score):", styles["SubHeaderStyle"]))
        story.append(Spacer(1, 0.1*inch))

        # Calculate total matches for win rate
        num_completions = len(completions_text)
        total_matches_per_completion = max(1, num_completions - 1)
        
        # Sort completions by primary score (descending)
        ranked_completions = sorted(completion_details, key=lambda x: x['score'], reverse=True)
        
        rank_table_data = [["Rank", "Score", "Wins", "Win Rate", "Image", "Original Index"]]
        rank_table_style = [('BACKGROUND', (0,0), (-1,0), colors.grey),
                            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                            ('ALIGN',(0,0),(-1,-1),'CENTER'),
                            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0,0), (-1,0), 12),
                            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
                            ('GRID',(0,0),(-1,-1),1,colors.black)]

        for rank, details in enumerate(ranked_completions):
            img_cell = None
            try:
                img_path = details['image_path']
                f = open(img_path, 'rb')
                opened_files.append(f)
                img_cell = ReportlabImage(f, width=1*inch, height=1*inch)
            except Exception as e:
                img_cell = Paragraph(f"Error: {e}", styles["CodeStyle"])
            win_rate = (details['wins'] / total_matches_per_completion) if total_matches_per_completion > 0 else 0
            rank_table_data.append([
                f"{rank+1}", 
                f"{details['score']:.4f}", 
                f"{details['wins']}", # Add wins column
                f"{win_rate:.1%}", # Add win rate column
                img_cell, 
                f"{details['index']+1}"
            ])
            
        rank_table = Table(rank_table_data, colWidths=[0.5*inch, 0.8*inch, 0.5*inch, 0.8*inch, 1.2*inch, 1*inch]) # Adjust colWidths
        rank_table.setStyle(TableStyle(rank_table_style))
        story.append(rank_table)
        story.append(Spacer(1, 0.3*inch))

        # --- PDF Logging: Pairwise Comparison Details ---
        if pairwise_results is not None:
            story.append(Paragraph("Pairwise Comparison Details (Conversational):", styles["SubHeaderStyle"]))
            story.append(Spacer(1, 0.1*inch))

            # Updated headers for conversational logging
            comparison_table_data = [["Comparison", "Image 1", "Image 2", "Final Verdict", "Conversation Log"]]
            comparison_table_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND',(0,1),(-1,-1),colors.lightblue),
                ('GRID',(0,0),(-1,-1),1,colors.black),
                ('VALIGN',(4,1), (4,-1), 'TOP'), # Align conversation log to top
                ('ALIGN',(4,1), (4,-1), 'LEFT'), # Align conversation log to left
            ]

            for result in pairwise_results:
                idx1 = result['comp_1_idx']
                idx2 = result['comp_2_idx']
                winner_idx = result['winner_idx']
                final_verdict_text = result.get('final_verdict', 'N/A') # Get final verdict
                conversation_log = result.get('conversation_log', {}) # Get conversation log

                # Format conversation log as a JSON string for the PDF cell
                try:
                    # Use html.escape to handle special characters within the JSON string
                    log_str_full = html.escape(json.dumps(conversation_log, indent=2))
                    # Truncate if too long to prevent LayoutError
                    max_log_length = 1000 # Adjust this character limit as needed
                    if len(log_str_full) > max_log_length:
                        log_str = log_str_full[:max_log_length] + "... (truncated)"
                    else:
                        log_str = log_str_full
                    # Replace newlines with <br/> for ReportLab Paragraph
                    # Use the new SmallCodeStyle for the log paragraph
                    log_paragraph = Paragraph(log_str.replace('\\n', '<br/>'), styles["SmallCodeStyle"])
                except Exception as e:
                    log_paragraph = Paragraph(f"Error formatting log: {e}", styles["SmallCodeStyle"]) # Use small style for errors too

                img1_cell = None
                try:
                    img1_path = train_model_image_paths[idx1]
                    f1 = open(img1_path, 'rb')
                    opened_files.append(f1)
                    img1_cell = ReportlabImage(f1, width=1*inch, height=1*inch)
                except Exception as e:
                    img1_cell = Paragraph(f"Error (Idx {idx1+1}): {e}", styles["CodeStyle"])
                   
                img2_cell = None
                try:
                    img2_path = train_model_image_paths[idx2]
                    f2 = open(img2_path, 'rb')
                    opened_files.append(f2)
                    img2_cell = ReportlabImage(f2, width=1*inch, height=1*inch)
                except Exception as e:
                    img2_cell = Paragraph(f"Error (Idx {idx2+1}): {e}", styles["CodeStyle"])

                # Determine winner text based on winner_idx (same logic as before)
                if winner_idx == idx1:
                    winner_paragraph = Paragraph(f"Image 1 (Idx {idx1+1})", styles["NormalLeft"])
                elif winner_idx == idx2:
                    winner_paragraph = Paragraph(f"Image 2 (Idx {idx2+1})", styles["NormalLeft"])
                else:
                    winner_paragraph = Paragraph(f"Tie/Error ({final_verdict_text})", styles["NormalLeft"])

                comparison_table_data.append([
                    f"Comp {idx1+1} vs Comp {idx2+1}",
                    img1_cell,
                    img2_cell,
                    Paragraph(html.escape(final_verdict_text), styles["NormalLeft"]), # Display final verdict
                    log_paragraph # Add formatted conversation log
                ])

            # Adjust column widths - make conversation log wider
            comparison_table = Table(comparison_table_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch, 3*inch]) 
            comparison_table.setStyle(TableStyle(comparison_table_style))
            story.append(comparison_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Force cleanup
            del comparison_table
            del comparison_table_data

    # --- Compute Loss (Run Every Iteration) ---
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        all_models["training_model"], all_models["base_model"], prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics (Original Logic)
    metrics.update(loss_metrics)

    # --- Build PDF and Cleanup ---
    try:
        print(f"Building PDF report for round {round_num}...")
        doc.build(story)
        print(f"Training report for round {round_num} saved to {pdf_path}")
    except Exception as e:
        print(f"ERROR building PDF for round {round_num}: {e}")
    finally:
        # Ensure all opened image files are closed
        print(f"Closing {len(opened_files)} image file handles...")
        for f in opened_files:
            try:
                f.close()
            except Exception as e_close:
                print(f"Warning: could not close file handle: {e_close}")
        del opened_files # Clear the list
        del story
        del doc
        torch.cuda.empty_cache()
        gc.collect()

    # --- Print Memory Usage ---
    try:
        with open('/proc/self/status') as f_status:
            for line in f_status:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    print(f"Current memory usage (RSS): {rss_kb / 1024:.2f} MB")
                    break
    except Exception as e_mem:
        print(f"Could not read memory usage: {e_mem}")
       
    # Return loss and metrics (Original Logic)
    return loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B", help="Name/path of base model")
    parser.add_argument("--judge_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Name of model to use as judge")
    parser.add_argument("--compare_model_name", type=str, default="gpt-4o-mini", help="Name of model to use for comparison")
    parser.add_argument("--dataset_name", type=str, default="debate", choices=["debate", "ld", "chopped", "svg"], help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="debate", choices=["debate", "ld", "chopped", "svg"], help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every N steps")
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
    parser.add_argument("--num_chains", type=int, default=6, help="Number of parallel generation chains")
    parser.add_argument("--eval_num_chains", type=int, default=2, help="Number of parallel generation chains for evaluation")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1280, help="Maximum completion length")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=6000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    args = parser.parse_args()
    return args


def force_gc():
    """
    Force aggressive garbage collection to free memory
    """
    for _ in range(3):  # Run multiple times to get all generations
        gc.collect()
    torch.cuda.empty_cache()

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
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    judge_model = llms.get_judge_model(args.judge_model_name, "cuda")
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

    for round_num in tqdm(range(start_round, args.num_train_iters), desc="Training Progress"):
        print(f"Round {round_num}")
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                all_models=all_models,
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Force garbage collection after evaluation
            force_gc()
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

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
        total_loss, train_metrics = grpo_loss(test_loader, train_loader, all_models, question, eval_class, device, round_num, train_log_dir, args)
        
        # Gradient accumulation
        total_loss = total_loss
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if True: #(round_num + 1) % args.gradient_accumulation_steps == 0:
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

        # Aggressive memory cleanup after each iteration
        del total_loss
        del train_metrics
        gc.collect()
        torch.cuda.empty_cache()
        
        # Print memory stats every 10 rounds if verbose
        if args.verbose and round_num % 10 == 0:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                    print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
                    print(f"GPU {i} max memory allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
    
