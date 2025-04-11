import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.backends.backend_pdf import PdfPages

def moving_average(data, window_size=5):
    """Calculate moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

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
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
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
            
            # Plot accuracy (if available)
            if all('overall_accuracy_percent' in eval_logs[step] for step in eval_steps):
                plt.figure(figsize=(12,7))
                # Use the correct key, already stored as percentage
                accuracy_values = [eval_logs[step]['overall_accuracy_percent'] for step in eval_steps] 
                plt.plot(eval_steps, accuracy_values, color='#2ecc71', linewidth=2.0, marker='o', label='Accuracy')
                plt.xlabel('Evaluation Round', fontsize=12) # Changed label
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Qwen 2.5 VL Oxford Flowers 102 Evaluation Accuracy', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.xticks(eval_steps) # Ensure ticks match evaluation rounds
                plt.tick_params(axis='x', labelsize=8) # Decrease x-axis label size
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                # Save as separate PNG file
                png_path = os.path.join(output_dir, 'evaluation_accuracy.png')
                plt.savefig(png_path, bbox_inches='tight')
                print(f"Evaluation accuracy plot saved to {png_path}")
                plt.close()
            else:
                print("Warning: 'overall_accuracy_percent' key missing in some evaluation logs. Skipping accuracy plot.")

            # Plot evaluation reward metrics (if available)
            # Get metric names from the first evaluation log's 'average_metrics' dict
            first_eval_step = eval_steps[0]
            if 'average_metrics' in eval_logs[first_eval_step] and isinstance(eval_logs[first_eval_step]['average_metrics'], dict):
                eval_metric_keys = list(eval_logs[first_eval_step]['average_metrics'].keys())
                print(f"Plotting evaluation reward metrics: {eval_metric_keys}")
                
                metric_color_map = {metric: color for metric, color in zip(eval_metric_keys, colors)} # Use defined colors

                for metric in eval_metric_keys:
                    color = metric_color_map.get(metric, '#34495e') # Default color
                    plt.figure(figsize=(12,7))
                    
                    # Access metrics within the correct nested dictionary
                    metric_values = []
                    valid_eval_steps = []
                    for step in eval_steps:
                        if 'average_metrics' in eval_logs[step] and metric in eval_logs[step]['average_metrics']:
                            metric_values.append(eval_logs[step]['average_metrics'][metric])
                            valid_eval_steps.append(step)
                        else:
                            print(f"Warning: Metric '{metric}' not found in average_metrics for round {step}. Skipping point.")
                    
                    if not valid_eval_steps:
                        print(f"Warning: No data found for evaluation metric '{metric}'. Skipping plot.")
                        plt.close()
                        continue

                    plot_title = metric.split("/")[-1].replace("_", " ").title()
                    plt.plot(valid_eval_steps, metric_values, color=color, linewidth=2.0, marker='o', label=plot_title)
                    plt.xlabel('Evaluation Round', fontsize=12) # Changed label
                    plt.ylabel(plot_title, fontsize=12)
                    plt.title(f'Evaluation: {plot_title}', fontsize=14, pad=20)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(valid_eval_steps) # Ensure ticks match evaluation rounds where data exists
                    plt.legend()
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
            else:
                print("Warning: 'average_metrics' key missing or not a dictionary in first evaluation log. Skipping evaluation reward plots.")
        else:
             print("No evaluation logs found or processed. Skipping evaluation plots.")

    print(f"Plots saved to {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and evaluation metrics from logs directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Main output directory containing training_logs and eval_logs subdirectories')
    # Removed log_dir as output_dir is now the standard
    args = parser.parse_args()
    plot_metrics(args.output_dir)
