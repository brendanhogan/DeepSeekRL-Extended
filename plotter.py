import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import re
from collections import defaultdict
from typing import Optional # Added for type hinting
import shutil

def moving_average(data, window_size=5):
    """Calculate moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Global text color (defined later in create_plots, but used by plot_metric)
text_color = '#E0E0E0' 
bg_color = '#212946'
grid_color = '#2A3459'

def apply_retro_futurism_style(ax, fig, line_color):
    """Applies a retro-futuristic style to the plot."""
    # bg_color, grid_color, text_color are now global for access in legend styling too
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.spines['top'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)

    ax.tick_params(axis='x', colors=text_color, labelsize=12) # Adjusted label size
    ax.tick_params(axis='y', colors=text_color, labelsize=12) # Adjusted label size
    ax.yaxis.label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.title.set_color(text_color)

    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.7)
    
    return line_color

def plot_metric(metric_name, rounds, values, output_path, line_color_hex, claude_score: Optional[float] = None):
    """Plots a single metric over rounds and saves it as a PNG."""
    global text_color, bg_color, grid_color # Ensure access to global style vars

    print(f"Plotting {metric_name}...")
    fig, ax = plt.subplots(figsize=(12, 7))

    actual_line_color = apply_retro_futurism_style(ax, fig, line_color_hex)

    # Main line for the trained model
    ax.plot(rounds, values, marker='o', linestyle='-', color=actual_line_color, linewidth=2, markersize=5, zorder=10, label='Trained Model')

    # Glow effect
    n_glow_lines = 10
    diff_linewidth = 1.0
    alpha_value = 0.04
    for n in range(1, n_glow_lines + 1):
        ax.plot(rounds, values, marker='', linestyle='-',
                linewidth=2 + (diff_linewidth * n),
                alpha=alpha_value,
                color=actual_line_color,
                zorder=5)

    ax.fill_between(rounds, values, color=actual_line_color, alpha=0.1, zorder=1)
    
    try:
        plt.rcParams['font.family'] = 'Consolas'
    except:
        print("Retro font not found, using default.")

    title_fontsize = 18
    label_fontsize = 14
    plot_title = f'{metric_name} Over Rounds'

    if metric_name == "avg_overall_metrics/click_hit_rate":
        plot_title = "Percent Correct Clicks"
        title_fontsize = 22
        label_fontsize = 16

    ax.set_title(plot_title, fontsize=title_fontsize, color=text_color, pad=20)
    ax.set_xlabel('Round Number', fontsize=label_fontsize, color=text_color)
    ax.set_ylabel('Metric Value', fontsize=label_fontsize, color=text_color)
    
    if all(isinstance(r, (int, np.integer)) for r in rounds):
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles=handles, labels=labels, loc='best', facecolor=bg_color, edgecolor=grid_color, fontsize=label_fontsize - 2, framealpha=0.8)
        for text_obj in legend.get_texts():
            text_obj.set_color(text_color)
            
    plt.tight_layout()
    try:
        fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150) # Increased DPI
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close(fig)

def create_plots():
    global text_color, bg_color, grid_color # Define globals for styling consistency
    text_color = '#E0E0E0'
    bg_color = '#212946'
    grid_color = '#2A3459'

    json_reports_dir = "captcha_run_full_f1only/eval_logs/json_reports/"
    plots_output_dir = "captcha_plots/"

    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)
        print(f"Created directory: {plots_output_dir}")

    all_metrics_data = defaultdict(lambda: {'rounds': [], 'values': []})
    json_files = [f for f in os.listdir(json_reports_dir) if f.startswith('average_scores_round_') and f.endswith('.json')]

    def extract_round_num(filename):
        match = re.search(r'average_scores_round_(\d+)\.json', filename)
        return int(match.group(1)) if match else -1

    json_files.sort(key=extract_round_num)
    
    if not json_files:
        print(f"No 'average_scores_round_*.json' files found in {json_reports_dir}")
        return
    print(f"Found {len(json_files)} average score JSON files for the main model.")

    for filename in json_files:
        round_num = extract_round_num(filename)
        if round_num == -1:
            continue
        file_path = os.path.join(json_reports_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for metric_name, value in data.items():
                if isinstance(value, (int, float)):
                    all_metrics_data[metric_name]['rounds'].append(round_num)
                    all_metrics_data[metric_name]['values'].append(value)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}, skipping.")

    if not all_metrics_data:
        print("No plottable data found in main model JSON files.")
        return

    line_colors = ['#08F7FE', '#FE53BB', '#F5D300', '#00FF41', '#FF6C11', '#FD1D53', '#710193', '#FFFFFF']

    for i, (metric_name, data) in enumerate(all_metrics_data.items()):
        if not data['rounds'] or not data['values']:
            print(f"Skipping metric {metric_name} due to missing rounds or values.")
            continue
        
        safe_metric_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', metric_name)
        output_filename = f"{safe_metric_name}.png"
        output_path = os.path.join(plots_output_dir, output_filename)
        current_line_color = line_colors[i % len(line_colors)]
        
        plot_metric(metric_name, data['rounds'], data['values'], output_path, current_line_color)

def plot_metrics(output_dir):
    """
    Plot training metrics from training_logs directory.
    Creates PDF with separate plots for each metric over training steps.
    Uses a modern, professional style with custom color palette.
    """
    # Load training logs
    train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
    if not os.path.exists(train_logs_path):
        print(f"Error: Training log file not found at {train_logs_path}")
        return
    with open(train_logs_path, 'r') as f:
        try:
            train_logs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {train_logs_path}: {e}")
            return

    # Load evaluation logs
    eval_logs = {}
    eval_json_dir = os.path.join(output_dir, 'eval_logs', 'json') # Updated path
    if os.path.exists(eval_json_dir):
        for filename in os.listdir(eval_json_dir):
            if filename.startswith('eval_metrics_') and filename.endswith('.json'):
                try:
                    # Corrected round number extraction
                    round_num = int(filename.split('_')[2].split('.')[0]) 
                    with open(os.path.join(eval_json_dir, filename), 'r') as f:
                        eval_logs[round_num] = json.load(f)
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not process evaluation log file {filename}: {e}")
    else:
        print(f"Warning: Evaluation log directory not found at {eval_json_dir}")


    # Set style and color palette
    plt.style.use('bmh')  # Using 'bmh' style which is a modern, clean style
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Ensure eval_logs/pdfs directory exists
    eval_pdfs_dir = os.path.join(output_dir, 'eval_logs', 'pdfs')
    os.makedirs(eval_pdfs_dir, exist_ok=True)
    print(f"Created directory: {eval_pdfs_dir}")
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    pdf_copy_path = os.path.join(eval_pdfs_dir, 'training_plots.pdf')
    
    with PdfPages(pdf_path) as pdf:
        
        # --- Plot Training Metrics ---
        
        # Collect all available metric keys from the first log entry
        if not train_logs:
            print("Warning: Training logs are empty. Skipping training plots.")
        else:
            first_step_key = list(train_logs.keys())[0]
            available_train_metrics = list(train_logs[first_step_key].keys())
            
            # Identify reward metrics (heuristic: key starts with 'rewards/' or is 'reward')
            reward_metrics = [m for m in available_train_metrics if m.startswith('rewards/') or m == 'reward']
            other_metrics = [m for m in available_train_metrics if m not in reward_metrics and m != 'learning_rate'] # Exclude LR for separate plot
            
            steps = sorted([int(x) for x in train_logs.keys()])

            # Plot identified reward metrics
            print(f"Plotting training reward metrics: {reward_metrics}")
            metric_color_map = {metric: color for metric, color in zip(reward_metrics, colors)}
            for metric in reward_metrics:
                color = metric_color_map.get(metric, '#34495e') # Default color if more metrics than colors
                plt.figure(figsize=(12,7))
                
                # Ensure metric exists across all steps, handling potential missing keys
                values = []
                valid_steps = []
                for step in steps:
                    step_str = str(step)
                    if metric in train_logs[step_str]:
                        values.append(train_logs[step_str][metric])
                        valid_steps.append(step)
                    else:
                        print(f"Warning: Metric '{metric}' not found for step {step}. Skipping this point.")

                if not valid_steps:
                    print(f"Warning: No data found for metric '{metric}'. Skipping plot.")
                    plt.close()
                    continue

                # Plot raw data with low alpha
                plt.plot(valid_steps, values, color=color, alpha=0.3, linewidth=1.5, label='Raw data')
                
                # Calculate and plot moving average if we have enough data points
                if len(values) >= 5: # Use >= 5 for moving average calculation
                    ma_values = moving_average(values, window_size=min(5, len(values))) # Adjust window if fewer points
                    # Adjust steps for moving average plot alignment
                    ma_steps = valid_steps[len(valid_steps)-len(ma_values):]
                    plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label='Moving average (window=5)')
                
                plot_title = metric.split("/")[-1].replace("_", " ").title()
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(plot_title, fontsize=12)
                plt.title(f'Training: {plot_title}', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

            # Plot learning rate (if available)
            if 'learning_rate' in available_train_metrics:
                print("Plotting learning rate...")
                plt.figure(figsize=(12,7))
                lr_values = [train_logs[str(step)]['learning_rate'] for step in steps if 'learning_rate' in train_logs[str(step)]]
                valid_lr_steps = [step for step in steps if 'learning_rate' in train_logs[str(step)]]
                if valid_lr_steps:
                    plt.plot(valid_lr_steps, lr_values, color='#e74c3c', linewidth=2.0, label='Learning Rate')
                    plt.xlabel('Training Steps', fontsize=12)
                    plt.ylabel('Learning Rate', fontsize=12)
                    plt.title('Learning Rate Schedule', fontsize=14, pad=20)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    pdf.savefig(bbox_inches='tight')
                else:
                     print("Warning: No learning rate data found.")
                plt.close()
            else:
                print("Warning: 'learning_rate' key not found in training logs.")


            # Plot other identified training metrics (e.g., loss, kl, reward_std)
            print(f"Plotting other training metrics: {other_metrics}")
            metric_color_map = {metric: color for metric, color in zip(other_metrics, ['#e67e22', '#9b59b6', '#3498db', '#f1c40f', '#1abc9c'])} # Different colors
            for metric in other_metrics:
                color = metric_color_map.get(metric, '#34495e')
                plt.figure(figsize=(12,7))

                values = []
                valid_steps = []
                for step in steps:
                     step_str = str(step)
                     if metric in train_logs[step_str]:
                        values.append(train_logs[step_str][metric])
                        valid_steps.append(step)
                     else:
                        print(f"Warning: Metric '{metric}' not found for step {step}. Skipping this point.")
                
                if not valid_steps:
                    print(f"Warning: No data found for metric '{metric}'. Skipping plot.")
                    plt.close()
                    continue

                plt.plot(valid_steps, values, color=color, alpha=0.3, linewidth=1.5, label=f'{metric} (Raw)')
                if len(values) >= 5:
                    ma_values = moving_average(values, window_size=min(5, len(values)))
                    ma_steps = valid_steps[len(valid_steps)-len(ma_values):]
                    plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label=f'{metric} (MA)')

                plot_title = metric.replace("_", " ").title()
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(plot_title, fontsize=12)
                plt.title(f'Training: {plot_title}', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

        # --- Plot Evaluation Metrics ---
        if eval_logs:
            print("Plotting evaluation metrics...")
            # Use round numbers as steps for evaluation plots
            eval_steps = sorted(eval_logs.keys()) 
            
            # Remove the dedicated accuracy plot section as it should be handled below if present in average_metrics
            # if all('overall_accuracy_percent' in eval_logs[step] for step in eval_steps):
            #     plt.figure(figsize=(12,7))
            #     # Use the correct key, already stored as percentage
            #     accuracy_values = [eval_logs[step]['overall_accuracy_percent'] for step in eval_steps] 
            #     plt.plot(eval_steps, accuracy_values, color='#2ecc71', linewidth=2.0, marker='o', label='Accuracy')
            #     plt.xlabel('Evaluation Round', fontsize=12) # Changed label
            #     plt.ylabel('Accuracy (%)', fontsize=12)
            #     plt.title('Evaluation Accuracy', fontsize=14, pad=20)
            #     plt.grid(True, alpha=0.3)
            #     plt.xticks(eval_steps) # Ensure ticks match evaluation rounds
            #     plt.legend()
            #     pdf.savefig(bbox_inches='tight')
            #     plt.close()
            # else:
            #     print("Warning: 'overall_accuracy_percent' key missing in some evaluation logs. Skipping accuracy plot.")

            # Plot all evaluation metrics found in 'average_metrics'
            if not eval_steps:
                 print("Warning: No evaluation rounds found.")
            else:
                 first_eval_step = eval_steps[0] # Get the first round to check for keys
                 # Check if the first log has the correct key and it's a dictionary
                 expected_key = 'average_metrics_per_example' # Use the correct key
                 if expected_key in eval_logs[first_eval_step] and isinstance(eval_logs[first_eval_step][expected_key], dict):
                    # Get all metric keys from the first round's dictionary
                    eval_metric_keys = list(eval_logs[first_eval_step][expected_key].keys())
                    print(f"Found evaluation metrics in '{expected_key}': {eval_metric_keys}")
                    
                    # Assign colors to each metric
                    metric_color_map = {metric: color for metric, color in zip(eval_metric_keys, colors)} 

                    for metric in eval_metric_keys:
                        color = metric_color_map.get(metric, '#34495e') # Default color if not enough colors
                        plt.figure(figsize=(12,7))
                        
                        # Extract metric values for all rounds, checking existence under the correct key
                        metric_values = []
                        valid_eval_steps = []
                        for step in eval_steps:
                            # Check if the key exists in this specific round's log and is a dict
                            if expected_key in eval_logs[step] and isinstance(eval_logs[step][expected_key], dict) and metric in eval_logs[step][expected_key]:
                                metric_values.append(eval_logs[step][expected_key][metric])
                                valid_eval_steps.append(step)
                            else:
                                print(f"Warning: Metric '{metric}' not found in {expected_key} for round {step}. Skipping point.")
                        
                        if not valid_eval_steps:
                            print(f"Warning: No data found for evaluation metric '{metric}'. Skipping plot.")
                            plt.close()
                            continue

                        # Generate title from metric key
                        plot_title = metric.replace("_", " ").replace("/", " - ").title() 
                        plt.plot(valid_eval_steps, metric_values, color=color, linewidth=2.0, marker='o', label=plot_title)
                        plt.xlabel('Evaluation Round', fontsize=12) 
                        plt.ylabel(plot_title, fontsize=12)
                        plt.title(f'Evaluation: {plot_title}', fontsize=14, pad=20)
                        plt.grid(True, alpha=0.3)
                        # Use MaxNLocator to reduce the number of x-axis ticks
                        ax = plt.gca() # Get current axes
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True)) 
                        plt.legend()
                        pdf.savefig(bbox_inches='tight')

                        # --- Save specific plots as PNG ---
                        print(plot_title)
                        if plot_title == "Avg Reward":
                             png_path = os.path.join(output_dir, "evaluation_avg_reward.png")
                             plt.savefig(png_path, bbox_inches='tight')
                             print(f"Saved specific plot: {png_path}")
                        elif plot_title == "Avg Metrics - Mean Abs Correlation Error":
                             png_path = os.path.join(output_dir, "evaluation_mean_abs_correlation_error.png")
                             plt.savefig(png_path, bbox_inches='tight')
                             print(f"Saved specific plot: {png_path}")
                        elif plot_title == "Metrics - F1 Score": # Added for F1 Score
                             png_path = os.path.join(output_dir, "evaluation_f1_score.png")
                             plt.savefig(png_path, bbox_inches='tight')
                             print(f"Saved specific plot: {png_path}")
                        # --- End save specific plots ---

                        plt.close()
                 else:
                    print(f"Warning: '{expected_key}' key missing or not a dictionary in first evaluation log (round {first_eval_step}). Skipping evaluation metric plots.")
        else:
             print("No evaluation logs found or processed. Skipping evaluation plots.")

    print(f"Plots saved to {pdf_path}")
    
    # Copy the PDF to eval_logs/pdfs directory
    try:
        shutil.copy2(pdf_path, pdf_copy_path)
        print(f"Also copied plots to {pdf_copy_path}")
    except Exception as e:
        print(f"Error copying PDF to eval_logs/pdfs directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and evaluation metrics from logs directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Main output directory containing training_logs and eval_logs subdirectories')
    # Removed log_dir as output_dir is now the standard
    args = parser.parse_args()
    plot_metrics(args.output_dir)
    create_plots()
    print("Plotting script finished.")
