import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional

import re

# Added for PDF generation
import json
from fpdf import FPDF

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



def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
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

        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} ####\n\n")
            f.write("RESPONSE:\n")
            f.write(gen['response'] + "\n\n")
            
            # Parse XML sections if present
            try:
                reasoning = gen['response'].split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                answer = gen['response'].split("<answer>\n")[1].split("\n</answer>")[0]
                f.write("PARSED SECTIONS:\n")
                f.write(f"Reasoning:\n{reasoning}\n")
                f.write(f"Answer:\n{answer}\n\n")
            except:
                f.write("ERROR: Could not parse XML sections\n\n")
            
            # Write reward scores
            f.write("REWARD SCORES:\n")
            for reward_name, reward_value in gen['scores'].items():
                f.write(f"{reward_name}: {reward_value:.4f}\n")
            # Total reward is sum of individual scores
            total_reward = sum(gen['scores'].values())
            f.write(f"Total reward: {total_reward:.4f}\n\n")
            f.write("-"*40 + "\n\n")


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


#############################
## PDF Generation Function ##
#############################

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'GRM Debate Evaluation Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10) 
        # Use multi_cell for better handling of long text and newlines
        self.multi_cell(0, 5, body)
        self.ln()

    def add_comparison(self, comparison_data):
        self.add_page()
        self.chapter_title(f"Comparison #{comparison_data.get('comparison_index', 'N/A')} (Example #{comparison_data.get('example_index', 'N/A')})")

        self.set_font('Arial', 'B', 10)
        self.cell(0, 5, f"Topic: {comparison_data.get('topic', 'N/A')}", 0, 1)
        self.ln(2)

        # Arguments
        self.set_font('Arial', 'BI', 10)
        self.cell(0, 5, "Argument 1 (Trained Model - Extracted):", 0, 1)
        self.set_font('Arial', '', 9) 
        self.multi_cell(0, 5, comparison_data.get('argument1_trained_extracted', 'N/A'))
        self.ln(1)
        
        self.set_font('Arial', 'BI', 10)
        self.cell(0, 5, "Argument 2 (Compare Model - Extracted):", 0, 1)
        self.set_font('Arial', '', 9) 
        self.multi_cell(0, 5, comparison_data.get('argument2_compare_extracted', 'N/A'))
        self.ln(3)

        # Rounds
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, "Evaluation Rounds:", 0, 1)
        self.ln(1)
        for round_detail in comparison_data.get('rounds', []):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, f"  Round {round_detail.get('round', 'N/A')}", 0, 1)
            
            if "error" in round_detail:
                self.set_font('Arial', 'I', 9)
                self.cell(0, 5, f"    Error: {round_detail['error']}", 0, 1)
                continue

            # Principles
            self.set_font('Arial', 'BI', 9)
            self.cell(0, 5, "    Principles:", 0, 1)
            self.set_font('Arial', '', 9)
            principles = round_detail.get('principles', [])
            for i, p in enumerate(principles):
                self.multi_cell(0, 5, f"      {i+1}. {p}")
            self.ln(1)

            # Arg1 Critiques/Scores
            self.set_font('Arial', 'BI', 9)
            self.cell(0, 5, f"    Argument 1 Critiques (Agg Score: {round_detail.get('arg1_aggregate_score', 'N/A')}):", 0, 1)
            self.set_font('Arial', '', 9)
            critiques1 = round_detail.get('arg1_critiques_scores', [])
            for c in critiques1:
                 self.multi_cell(0, 5, f"      Principle: {c.get('principle', 'N/A')}")
                 self.multi_cell(0, 5, f"      Critique: {c.get('critique', 'N/A')}")
                 self.multi_cell(0, 5, f"      Score: {c.get('score', 'N/A')}")
                 self.ln(1)

            # Arg2 Critiques/Scores
            self.set_font('Arial', 'BI', 9)
            self.cell(0, 5, f"    Argument 2 Critiques (Agg Score: {round_detail.get('arg2_aggregate_score', 'N/A')}):", 0, 1)
            self.set_font('Arial', '', 9)
            critiques2 = round_detail.get('arg2_critiques_scores', [])
            for c in critiques2:
                 self.multi_cell(0, 5, f"      Principle: {c.get('principle', 'N/A')}")
                 self.multi_cell(0, 5, f"      Critique: {c.get('critique', 'N/A')}")
                 self.multi_cell(0, 5, f"      Score: {c.get('score', 'N/A')}")
                 self.ln(1)

            # Round Winner
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, f"    Round Winner: {round_detail.get('round_winner', 'N/A')}", 0, 1)
            self.ln(2) # Space between rounds
        
        # Overall Winner
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, f"Overall Winner: {comparison_data.get('overall_winner', 'N/A')}", 0, 1)
        self.ln(2)

def create_evaluation_pdf(
    detailed_log_path: str,
    summary_metrics_path: str,
    win_rate_plot_path: Optional[str],
    output_pdf_path: str
) -> None:
    """
    Generates a PDF report from evaluation log files.

    Args:
        detailed_log_path: Path to the JSON file containing detailed logs.
        summary_metrics_path: Path to the JSON file containing summary metrics.
        output_pdf_path: Path where the output PDF should be saved.
        win_rate_plot_path: Optional path to the win rate plot PNG.
    """
    try:
        with open(detailed_log_path, 'r') as f:
            detailed_logs = json.load(f)
    except Exception as e:
        print(f"Error loading detailed log file {detailed_log_path}: {e}")
        return

    try:
        with open(summary_metrics_path, 'r') as f:
            summary_metrics = json.load(f)
    except Exception as e:
        print(f"Error loading summary metrics file {summary_metrics_path}: {e}")
        summary_metrics = {} # Continue without summary if file fails

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.add_page()

    # Add Summary Section First
    pdf.chapter_title("Overall Evaluation Summary")
    summary_data = summary_metrics.get('metrics', summary_metrics) # Handle potential nesting
    win_rate = summary_data.get('win_rate', 'N/A')
    total_wins = summary_data.get('total_wins', 'N/A')
    total_comparisons = summary_data.get('total_comparisons', 'N/A')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, f"Win Rate: {win_rate:.2f}%" if isinstance(win_rate, (int, float)) else f"Win Rate: {win_rate}", 0, 1)
    pdf.cell(0, 6, f"Total Wins / Comparisons: {total_wins} / {total_comparisons}", 0, 1)
    pdf.ln(4)

    avg_scores = summary_data.get('average_scores', {})
    if avg_scores:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, "Average Format Scores:", 0, 1)
        pdf.set_font('Arial', '', 10)
        for k, v in avg_scores.items():
             pdf.cell(0, 5, f"  {k.replace('rewards/','')}: {v:.4f}", 0, 1)
        pdf.ln(5)
    
    # Add Win Rate Plot if path provided and file exists
    if win_rate_plot_path and os.path.exists(win_rate_plot_path):
        try:
             pdf.image(win_rate_plot_path, x = pdf.get_x(), y = pdf.get_y(), w = pdf.w - pdf.l_margin - pdf.r_margin) # Adjust width to fit page margins
             pdf.ln(10) # Add some space after the image
        except Exception as e:
             print(f"Error embedding win rate plot {win_rate_plot_path}: {e}")

    # Add detailed comparisons
    pdf.chapter_title("Detailed Debate Comparisons")
    if not detailed_logs:
        pdf.chapter_body("No detailed logs were found or provided.")
    else:
        for comparison_data in detailed_logs:
            pdf.add_comparison(comparison_data)

    try:
        pdf.output(output_pdf_path, 'F')
        print(f"Successfully generated PDF report: {output_pdf_path}")
    except Exception as e:
        print(f"Error generating PDF report {output_pdf_path}: {e}")
