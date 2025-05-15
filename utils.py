import re
import os
import json
import html
import torch
import random
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from reportlab.lib.units import inch
from typing import Any, Dict, Optional, Callable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from PIL import Image as PILImage, ImageDraw # To avoid conflict with ReportLabImage, explicitly import ImageDraw
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qwen_vl_utils import process_vision_info

import evaluator
from gui_generator import GUIGenerator # For plot_predictions type hint if GUI specific logic

# Import constants from captcha_generator if needed for plotting logic
from captcha_generator import FINAL_DIM as CAPTCHA_FINAL_DIM, GRID_SIZE as CAPTCHA_GRID_SIZE, BANNER_ABS_HEIGHT as CAPTCHA_BANNER_ABS_HEIGHT, PADDING_SIZE as CAPTCHA_PADDING_SIZE, CELL_DIM as CAPTCHA_CELL_DIM

MAX_COMPLETIONS_PER_PAGE_PDF = 2
MAX_PROMPT_LENGTH_PDF = 300 # Add the missing constant definition
MAX_ANSWER_LENGTH_PDF = 200
MAX_COMPLETION_LENGTH_PDF = 500

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add set_seed (alias for seed_everything)
set_seed = seed_everything
def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: Dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, 'a') as f: # Append mode
        f.write(json.dumps(log_data, indent=2) + "\n---\n") # Add separator


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

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

def get_per_token_logps_vl(model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt):
    """
    We have the input ids - all the correct tokens including all chate templates/special tokens etc 
    We just need to include the image - and have the same sort of obj to pass to the model to generate 
    the logits 

    So lets generate with the image


    resulting to a very non-generic way to do this - TODO: make this better
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

    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")  
    image_inputs, video_inputs = process_vision_info(conversation)

    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left"
    ).to(model.device).to(model.dtype)

    # Repeat input tensors for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(input_ids.shape[0], *([1] * (value.dim() - 1)))
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value 

    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


########################
## PDF/HTML/TEXT STUFF ##
########################

# Constants for PDF generation
PDF_IMAGE_WIDTH = 400  # Max width for images in points (adjust as needed)
PDF_MAX_IMAGE_HEIGHT = 550 # Max height for images in points (adjust as needed)

def _extract_tagged_content(text: str, tag: str) -> Optional[str]:
    """Extracts content within a specific XML-like tag."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def _setup_training_log_directory(output_dir: str) -> str:
    """Creates and returns the path for the training log directory."""
    training_log_dir = os.path.join(output_dir, "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)
    return training_log_dir

def _setup_eval_directories(base_output_dir: str) -> tuple[str, str, str]:
    """Creates and returns paths for evaluation log directories (PDF and JSON)."""
    logs_dir = os.path.join(base_output_dir, 'eval_logs')
    pdf_dir = os.path.join(logs_dir, 'pdfs')
    json_dir = os.path.join(logs_dir, 'json_reports')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    return logs_dir, pdf_dir, json_dir

def _setup_pdf(pdf_path: str) -> tuple[SimpleDocTemplate, dict, list]:
    """Initializes the PDF document, styles, and story list."""
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [] 
    styles['Code'].fontSize = 10 
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold'))
    # Add a justified style for longer text blocks
    styles.add(ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=TA_JUSTIFY))
    return doc, styles, story

def _truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

def _add_example_header_to_pdf(story: list, styles: dict, image_path: str, 
                               prompt_text: str, answer_data: Any, example_num: int, dataset_type: str,
                               is_hard: bool = False):
    title = f"Example {example_num + 1}"
    if is_hard:
        title += " (Hard Subset)"
    story.append(Paragraph(title, styles['h2']))
    
    # Display image if path is valid
    if os.path.exists(image_path):
        try:
            img = PILImage.open(image_path)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.5 * inch # Max width for the image in PDF
            display_height = display_width * aspect
            # Cap height to prevent overly tall images
            if display_height > 3.5 * inch:
                display_height = 3.5 * inch
                display_width = display_height / aspect
            
            # For CAPTCHA, add a header before the input image
            if dataset_type == 'captcha':
                story.append(Paragraph("<b>Input CAPTCHA Image:</b>", styles['BodyText']))
            
            story.append(PlatypusImage(image_path, width=display_width, height=display_height))
            story.append(Spacer(1, 0.1*inch))
            
            # For CAPTCHA, also display the solution image if available
            if dataset_type == 'captcha' and answer_data and 'solution_image_path' in answer_data:
                solution_image_path = answer_data['solution_image_path']
                if os.path.exists(solution_image_path):
                    try:
                        story.append(Paragraph("<b>Ground Truth Solution Image:</b>", styles['BodyText']))
                        sol_img = PILImage.open(solution_image_path)
                        sol_img_width, sol_img_height = sol_img.size
                        sol_aspect = sol_img_height / float(sol_img_width)
                        sol_display_width = 2.5 * inch
                        sol_display_height = sol_display_width * sol_aspect
                        if sol_display_height > 3.5 * inch:
                            sol_display_height = 3.5 * inch
                            sol_display_width = sol_display_height / sol_aspect
                        
                        story.append(PlatypusImage(solution_image_path, width=sol_display_width, height=sol_display_height))
                        story.append(Spacer(1, 0.1*inch))
                    except Exception as e:
                        story.append(Paragraph(f"Error loading solution image: {solution_image_path}. Error: {e}", styles['BodyText']))
                else:
                    story.append(Paragraph(f"Solution image not found: {solution_image_path}", styles['BodyText']))
                    
        except Exception as e:
            story.append(Paragraph(f"Error loading image: {image_path}. Error: {e}", styles['BodyText']))
    else:
        story.append(Paragraph(f"Image not found: {image_path}", styles['BodyText']))

    # Display full prompt without truncation, escape HTML tags
    story.append(Paragraph(f"<b>Prompt:</b>", styles['BodyText']))
    escaped_prompt = html.escape(prompt_text).replace('\n', '<br/>')
    story.append(Paragraph(escaped_prompt, styles['Code']))
    
    # Display answer/target information based on dataset type
    if dataset_type == 'gui':
        target_name = answer_data.get('name', 'N/A')
        target_bbox = answer_data.get('bounding_box', 'N/A')
        story.append(Paragraph(f"<b>Target Object:</b> {target_name}", styles['BodyText']))
        story.append(Paragraph(f"<b>Target BBox:</b> {str(target_bbox)}", styles['BodyText']))
    elif dataset_type == 'captcha':
        target_class_name = answer_data.get('target_class_name', 'N/A')
        article = "an" if target_class_name.lower()[0] in 'aeiou' else "a"
        story.append(Paragraph(f"<b>Target:</b> Select all squares with {article} {target_class_name}", styles['BodyText']))
        # Optionally, display the boolean list or a simple text representation if helpful
        # story.append(Paragraph(f"<b>Target Squares (boolean):</b> {str(answer_data.get('target_squares_boolean'))}", styles['Code']))
    else: # Clock, Correlation
        story.append(Paragraph(f"<b>Ground Truth Answer:</b> {_truncate_text(str(answer_data), MAX_ANSWER_LENGTH_PDF)}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))

def _add_completion_to_pdf(story: list, styles: dict, completion_text: str, 
                           metrics: Optional[Dict[str, Any]], completion_idx: int, 
                           dataset_type: str,
                           image_path_for_completion_pdf: Optional[str] = None,
                           captcha_data: Optional[Dict[str, Any]] = None): 
    story.append(Paragraph(f"Completion {completion_idx + 1}", styles['h3']))

    # Display image for this completion (e.g., with click for GUI)
    if image_path_for_completion_pdf and os.path.exists(image_path_for_completion_pdf):
        try:
            img = PILImage.open(image_path_for_completion_pdf)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.0 * inch # Slightly smaller for completion image
            display_height = display_width * aspect
            if display_height > 3.0 * inch:
                display_height = 3.0 * inch
                display_width = display_height / aspect
            
            story.append(PlatypusImage(image_path_for_completion_pdf, width=display_width, height=display_height))
            story.append(Spacer(1, 0.05*inch))
        except Exception as e:
            story.append(Paragraph(f"Error loading completion image: {image_path_for_completion_pdf}. Error: {e}", styles['Italic']))
    
    # Display full completion text (escaped)
    story.append(Paragraph(f"<b>Full Model Output:</b>", styles['BodyText']))
    escaped_completion = html.escape(completion_text).replace('\n', '<br/>')
    story.append(Paragraph(escaped_completion, styles['Code']))
    story.append(Spacer(1, 0.05*inch))

    # Extract and display reasoning and answer separately
    reasoning = _extract_tagged_content(completion_text, 'reasoning')
    answer = _extract_tagged_content(completion_text, 'answer')

    story.append(Paragraph(f"<b>Extracted Reasoning:</b>", styles['BodyText']))
    story.append(Paragraph(html.escape(reasoning) if reasoning else "<i>N/A</i>", styles['Code']))
    story.append(Spacer(1, 0.05*inch))

    story.append(Paragraph(f"<b>Extracted Answer:</b>", styles['BodyText']))
    story.append(Paragraph(html.escape(answer) if answer else "<i>N/A</i>", styles['Code']))
    story.append(Spacer(1, 0.1*inch))

    # Add CAPTCHA-specific plain English explanation if dataset_type is 'captcha' and captcha_data is provided
    if dataset_type == 'captcha' and captcha_data is not None:
        true_positives = captcha_data.get('true_positives', 0)
        false_positives = captcha_data.get('false_positives', 0)
        false_negatives = captcha_data.get('false_negatives', 0)
        total_targets = captcha_data.get('total_targets', 0)
        total_clicks = captcha_data.get('total_clicks', 0)
        
        # Create plain English explanation
        accuracy_text = [
            f"<b>CAPTCHA Accuracy Summary:</b>",
            f"• Model clicked {true_positives} out of {total_targets} correct targets ({true_positives/total_targets*100:.1f}% recall)" if total_targets > 0 else "• No correct targets to click",
            f"• Model made {total_clicks} total clicks",
        ]
        
        if false_positives > 0:
            accuracy_text.append(f"• Overclicked by {false_positives} (clicked {false_positives} incorrect squares)")
        
        if false_negatives > 0:
            accuracy_text.append(f"• Missed {false_negatives} correct targets")
            
        # Add precision information
        if total_clicks > 0:
            precision = true_positives / total_clicks
            accuracy_text.append(f"• Precision: {precision*100:.1f}% of clicks were on correct targets")
            
        # Join with line breaks and add to PDF
        story.append(Paragraph("<br/>".join(accuracy_text), styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))

    if metrics:
        # Create a more structured table for metrics if many, or simple paragraphs
        metrics_data = [["Metric", "Value"]]
        # Retrieve total_reward_this_completion from metrics if it exists
        total_reward_val = metrics.pop('total_reward_this_completion', None) 

        for name, value in metrics.items():
            # Optionally shorten name for display
            display_name = name.replace("rewards/", "").replace("metrics/", "")
            if isinstance(value, float):
                metrics_data.append([display_name, f"{value:.4f}"])
            else:
                metrics_data.append([display_name, str(value)])
        
        if total_reward_val is not None:
             metrics_data.append(["Total Reward (this completion)", f"{total_reward_val:.4f}"])
        
        if len(metrics_data) > 1:
             # Adjusted colWidths: more space for metric name
            table = Table(metrics_data, colWidths=[2.8*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), # Improve vertical alignment
                ('WORDWRAP', (0,1), (0,-1)), # Allow metric names to wrap
            ]))
            story.append(table)
    story.append(Spacer(1, 0.15*inch))

def _process_single_completion_for_eval(
    completion_text: str, 
    eval_class: evaluator.RewardEvaluator, 
    answer_data: Any, # This is target_details for GUI, answer_str for clock/correlation
    device: str, 
    story: Optional[list] = None, # Make it optional
    styles: Optional[dict] = None, # Make it optional
    completion_idx: int = 0,
    dataset_type: str = 'gui',
    original_image_path: Optional[str] = None,
    vis_image_path_for_pdf: Optional[str] = None,
    gui_plotter: Optional[callable] = None,
    verbose: bool = False # Added verbose for plot_captcha_evaluation
) -> Optional[dict[str, float]]:
    """
    Processes a single completion text for evaluation and optionally adds it to a PDF story.
    Returns a dictionary of metric scores for this single completion.
    """
    if dataset_type == 'gui' or dataset_type == 'captcha': # Captcha also uses dict answer_data
        current_answers_list = [answer_data]
        current_completions_list = [[{'content': completion_text}]]
    else: # clock, correlation
        # For clock/correlation, answer_data is the answer string.
        current_answers_list = [answer_data]
        current_completions_list = [[{'content': completion_text}]]

    # Get rewards and metrics for this single completion
    # The evaluator's compute_rewards should return rewards_per_func and metrics
    # rewards_per_func will be a tensor for this one completion, e.g., shape [1, num_reward_components]
    # metrics will be a dict like {'metric_name': value_tensor}
    rewards_per_func_single, metrics_single_dict_tensors = eval_class.compute_rewards(
        prompts=None, # Not always needed by evaluator if context is in answer_data
        completions=current_completions_list,
        answers=current_answers_list,
        device=device # device might be used by evaluator internally
    )

    # Convert metric tensors to scalar floats and ensure all are included
    # Also, get the reward breakdown for PDF logging if story is present
    processed_metrics_for_return = {}
    if metrics_single_dict_tensors and isinstance(metrics_single_dict_tensors, dict):
        for k, v_tensor in metrics_single_dict_tensors.items():
            if torch.is_tensor(v_tensor) and v_tensor.numel() == 1:
                processed_metrics_for_return[k] = v_tensor.item()
            elif isinstance(v_tensor, (float, int)):
                 processed_metrics_for_return[k] = v_tensor
            # else: might be other types of metrics not directly plottable/averageable

    # Get reward breakdown if needed (e.g., for PDF)
    # rewards_per_func_single should be for one sample, e.g., shape [1, num_reward_components]
    # We need to pass the actual reward scores for this one completion to get_reward_breakdown
    reward_scores_for_breakdown = rewards_per_func_single[0] # Get the tensor for the first (only) sample
    
    # Add overall reward to the metrics
    total_reward_single = reward_scores_for_breakdown.sum().item()
    processed_metrics_for_return['reward'] = total_reward_single # Ensure 'reward' key exists

    # --- PDF Logging Section ---
    # Only attempt PDF operations if story and styles are provided
    if story is not None and styles is not None:
        reward_breakdown_for_pdf = eval_class.get_reward_breakdown(reward_scores_for_breakdown)
        
        # Add total_reward_single to the dictionary that will be passed as metrics to _add_completion_to_pdf
        reward_breakdown_for_pdf['total_reward_this_completion'] = total_reward_single
        
        img_path_for_pdf_entry = None
        captcha_stats = None
        
        if dataset_type == 'gui' and original_image_path and vis_image_path_for_pdf and gui_plotter:
            try:
                # Plot click for GUI task if a plotter is provided
                if isinstance(eval_class, evaluator.GUIEvaluator): # Check if it's the right evaluator
                    parsed_click = eval_class._extract_coordinates(completion_text) # Protected access, but used in main.py
                    if parsed_click:
                        pil_img = PILImage.open(original_image_path)
                        plot_data = [{
                            "name": "VLM Click", 
                            "center_x": parsed_click[0], 
                            "center_y": parsed_click[1],
                            "is_truth": False 
                        }]
                        # GUIGenerator.plot_predictions returns a PIL Image
                        img_w_click = gui_plotter(pil_img, plot_data, pred_color="red")
                        img_w_click.save(vis_image_path_for_pdf)
                        img_path_for_pdf_entry = vis_image_path_for_pdf
                    else:
                        # If click not parsed, use original image for PDF (or None if vis_image_path_for_pdf was for specific click)
                        img_path_for_pdf_entry = original_image_path # Or handle as per main.py logic
                else:
                    img_path_for_pdf_entry = original_image_path # Fallback for non-GUIEvaluator or if no click
            except Exception as plot_err:
                if verbose: # Assuming verbose is accessible or passed
                    print(f"  Warning: Error plotting click for PDF (utils): {plot_err}")
                img_path_for_pdf_entry = original_image_path # Fallback
        elif dataset_type == 'captcha' and original_image_path and vis_image_path_for_pdf:
            # CAPTCHA specific PDF logging for evaluation
            img_path_for_pdf_entry = None
            
            # Create a temporary CaptchaEvaluator to extract clicks
            from evaluator import CaptchaEvaluator
            temp_captcha_evaluator = CaptchaEvaluator()
            predicted_clicks = temp_captcha_evaluator._extract_click_calls(completion_text)
            
            # Calculate CAPTCHA accuracy stats for the plain English summary
            if 'target_squares_boolean' in answer_data:
                target_squares_boolean = answer_data['target_squares_boolean']
                total_targets = sum(1 for x in target_squares_boolean if x)
                total_clicks = len(predicted_clicks)
                
                # Manually calculate TP, FP, FN for clarity
                tp = 0  # True positives
                fp = 0  # False positives
                clicked_squares = set()
                
                # For each click, determine which square it falls in and if it's a target
                for px, py in predicted_clicks:
                    # Convert pixel coordinates to grid cell
                    # Calculate grid parameters using constants from CAPTCHA generator
                    content_width_unpadded = CAPTCHA_CELL_DIM * CAPTCHA_GRID_SIZE
                    content_height_unpadded = CAPTCHA_BANNER_ABS_HEIGHT + content_width_unpadded
                    pre_resize_width = content_width_unpadded + 2 * CAPTCHA_PADDING_SIZE
                    pre_resize_height = content_height_unpadded + 2 * CAPTCHA_PADDING_SIZE
                    scale_x = CAPTCHA_FINAL_DIM / pre_resize_width
                    scale_y = CAPTCHA_FINAL_DIM / pre_resize_height
                    final_grid_start_x = CAPTCHA_PADDING_SIZE * scale_x
                    final_grid_start_y = (CAPTCHA_PADDING_SIZE + CAPTCHA_BANNER_ABS_HEIGHT) * scale_y
                    final_cell_width = CAPTCHA_CELL_DIM * scale_x
                    final_cell_height = CAPTCHA_CELL_DIM * scale_y
                    
                    # Determine grid position
                    if (px < final_grid_start_x or py < final_grid_start_y or 
                        px >= final_grid_start_x + final_cell_width * CAPTCHA_GRID_SIZE or 
                        py >= final_grid_start_y + final_cell_height * CAPTCHA_GRID_SIZE):
                        # Click is outside the grid
                        fp += 1
                        continue
                    
                    # Calculate which cell was clicked
                    col = int((px - final_grid_start_x) / final_cell_width)
                    row = int((py - final_grid_start_y) / final_cell_height)
                    col = max(0, min(col, CAPTCHA_GRID_SIZE - 1))
                    row = max(0, min(row, CAPTCHA_GRID_SIZE - 1))
                    
                    square_idx = row * CAPTCHA_GRID_SIZE + col
                    
                    # Only count unique square clicks (first click per square)
                    if square_idx not in clicked_squares:
                        clicked_squares.add(square_idx)
                        # Check if it's a target square
                        if square_idx < len(target_squares_boolean) and target_squares_boolean[square_idx]:
                            tp += 1
                        else:
                            fp += 1
                
                # Calculate false negatives (missed targets)
                fn = total_targets - tp
                
                # Store the stats for the plain English explanation
                captcha_stats = {
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'total_targets': total_targets,
                    'total_clicks': total_clicks
                }
            
            # Plot the visualization with the clicks
            plot_captcha_evaluation(
                base_image_path=original_image_path,
                predicted_clicks=predicted_clicks,
                target_squares_boolean=answer_data['target_squares_boolean'],
                output_path=vis_image_path_for_pdf,
                verbose=verbose)
            img_path_for_pdf_entry = vis_image_path_for_pdf
                
        elif dataset_type != 'gui' and original_image_path:
             img_path_for_pdf_entry = original_image_path

        # Add the completion to the PDF with the appropriate image and CAPTCHA stats
        _add_completion_to_pdf(
            story, styles, completion_text, 
            metrics=reward_breakdown_for_pdf, # Pass the breakdown as metrics
            completion_idx=completion_idx,
            dataset_type=dataset_type,
            image_path_for_completion_pdf=img_path_for_pdf_entry, # Pass the plotted image
            captcha_data=captcha_stats # Pass the captcha statistics
        )
        
    # --- End PDF Logging Section ---

    return processed_metrics_for_return

def _calculate_and_log_final_metrics(all_avg_scores: dict, json_dir: str, round_num: int, verbose: bool):
    """Saves combined average scores (overall, normal, hard) to JSON and prints if verbose."""
    # Save average scores to a JSON file
    avg_scores_path = os.path.join(json_dir, f'average_scores_round_{round_num}.json')
    with open(avg_scores_path, 'w') as f:
        # Log all average scores (overall, normal, hard)
        json.dump(all_avg_scores, f, indent=4)
    
    if verbose:
        print(f"\n--- Evaluation Results (Round {round_num}) ---")
        print(f"Average scores saved to {avg_scores_path}")
        print("Average Scores Breakdown:")
        # Nicely print the different groups if they exist
        print("  --- Overall --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_overall_"):
                 print(f"    {name.replace('avg_overall_',''):<35}: {value:.4f}")
        print("  --- Normal Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_normal_"):
                 print(f"    {name.replace('avg_normal_',''):<35}: {value:.4f}")
        if not any(k.startswith("avg_normal_") for k in all_avg_scores):
            print("    (No normal examples in this evaluation)")
        print("  --- Hard Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_hard_"):
                 print(f"    {name.replace('avg_hard_',''):<35}: {value:.4f}")
        if not any(k.startswith("avg_hard_") for k in all_avg_scores):
            print("    (No hard examples in this evaluation)")
        print("-" * 40)

def _add_training_completion_to_pdf(story: list, styles: dict, 
                                    completion_text: str, 
                                    reward_breakdown: Dict[str, float], 
                                    advantage: float, 
                                    completion_idx: int, 
                                    dataset_type: str, # Added dataset_type for consistency
                                    image_path_for_completion_pdf: Optional[str] = None):
    """Adds details for a single training completion (including scores and advantage) to the PDF story."""
    story.append(Paragraph(f"Training Completion {completion_idx + 1}", styles['h3']))

    # Display visualized image (e.g., with click for GUI)
    if image_path_for_completion_pdf and os.path.exists(image_path_for_completion_pdf):
        try:
            img = PILImage.open(image_path_for_completion_pdf)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.0 * inch 
            display_height = display_width * aspect
            if display_height > 3.0 * inch:
                display_height = 3.0 * inch
                display_width = display_height / aspect
            
            story.append(PlatypusImage(image_path_for_completion_pdf, width=display_width, height=display_height))
            story.append(Spacer(1, 0.05*inch))
        except Exception as e:
            story.append(Paragraph(f"Error loading training completion image: {image_path_for_completion_pdf}. Error: {e}", styles['Italic']))
    
    # Display full completion text (escaped)
    story.append(Paragraph(f"<b>Full Model Output:</b>", styles['BodyText']))
    escaped_completion = html.escape(completion_text).replace('\n', '<br/>')
    story.append(Paragraph(escaped_completion, styles['Code']))
    story.append(Spacer(1, 0.05*inch))

    # Display Reward Breakdown and Advantage
    story.append(Paragraph(f"<b>Scores & Advantage:</b>", styles['BodyText']))
    data_table = [["Component", "Value"]]
    total_reward = 0
    if reward_breakdown:
        for name, value in reward_breakdown.items():
            data_table.append([name, f"{value:.4f}"])
            total_reward += value
    data_table.append(["Total Reward (sum)", f"{total_reward:.4f}"])
    data_table.append(["Advantage", f"{advantage:.4f}"])
    
    table = Table(data_table, colWidths=[2.0*inch, 1.5*inch]) # Adjusted width
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP', (0,1), (0,-1)),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15*inch))

def plot_captcha_evaluation(
    base_image_path: str,
    predicted_clicks: list[tuple[int, int]],
    target_squares_boolean: list[bool],
    output_path: str,
    verbose: bool = False
) -> bool:
    """Just plots X's where the model clicked."""
    img = PILImage.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw an X for each click
    for x, y in predicted_clicks:
        # Draw X with fixed size
        size = 10
        draw.line([(x - size, y - size), (x + size, y + size)], fill=(255, 0, 0), width=3)
        draw.line([(x + size, y - size), (x - size, y + size)], fill=(255, 0, 0), width=3)
    
    img.save(output_path)
    return True




