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
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    # Load training logs
    train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
    if not os.path.exists(train_logs_path):
        raise FileNotFoundError(f"Training log file not found at expected path: {train_logs_path}")
    with open(train_logs_path, 'r') as f:
        train_logs = json.load(f)

    # Load evaluation logs
    eval_logs = {}
    eval_logs_dir = os.path.join(output_dir, 'eval_logs')
    if os.path.exists(eval_logs_dir): # Check if eval logs directory exists
        for filename in os.listdir(eval_logs_dir):
            if filename.startswith('metrics_') and filename.endswith('.json'):
                step = int(filename.split('_')[1].split('.')[0])
                try:
                    with open(os.path.join(eval_logs_dir, filename), 'r') as f:
                        data = json.load(f)
                        # Adjust structure: main.py nests metrics under 'metrics' key
                        # and uses 'accuracy' key for win_rate in debate scenarios
                        eval_logs[step] = data 
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}")
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")

    # Set style and color palette
    plt.style.use('bmh')
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    with PdfPages(pdf_path) as pdf:
        
        # Determine which reward metrics exist in the logs
        sample_metrics = next(iter(train_logs.values()))
        reward_metrics = [key for key in sample_metrics.keys() if key.startswith('rewards/')]
        
        # Plot reward metrics
        for metric, color in zip(reward_metrics, colors):
            plt.figure(figsize=(12,7))
            steps = [int(x) for x in train_logs.keys()]
            values = [metrics.get(metric, 0) for metrics in train_logs.values()]
            
            # Plot raw data with low alpha
            plt.plot(steps, values, color=color, alpha=0.3, linewidth=1.5, label='Raw data')
            
            # Calculate and plot moving average if we have enough data points
            if len(values) > 5:
                ma_values = moving_average(values)
                ma_steps = steps[len(steps)-len(ma_values):]
                plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label='Moving average')
            
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel(f'{metric.split("/")[-1].replace("_", " ").title()}', fontsize=12)
            plt.title(f'{metric.split("/")[-1].replace("_", " ").title()}', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Plot other metrics that exist in both evaluators
        common_metrics = ['learning_rate', 'reward_std', 'loss', 'kl']
        metric_colors = {
            'learning_rate': '#e74c3c',
            'reward_std': '#3498db',
            'loss': '#e67e22',
            'kl': '#9b59b6'
        }
        
        for metric in common_metrics:
            if any(metric in metrics for metrics in train_logs.values()):
                plt.figure(figsize=(12,7))
                steps = [int(x) for x in train_logs.keys()]
                values = [metrics.get(metric, 0) for metrics in train_logs.values()]

                plt.plot(steps, values, color=metric_colors[metric], alpha=0.3, linewidth=1.5, label=f'{metric} (Raw)')
                if len(values) > 5:
                    ma_values = moving_average(values)
                    ma_steps = steps[len(steps)-len(ma_values):]
                    plt.plot(ma_steps, ma_values, color=metric_colors[metric], linewidth=2.5, label=f'{metric} (MA)')

                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                plt.title(f'{metric.replace("_", " ").title()}', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

        # Plot evaluation metrics if they exist
        if eval_logs:
            eval_steps = sorted(eval_logs.keys())
            
            # Check the structure of the first log entry to determine metric key
            first_log_entry = eval_logs[eval_steps[0]]
            
            # Plot win rate (stored under 'accuracy' key from main.py eval) or accuracy
            plt.figure(figsize=(12,7))
            # Check if 'accuracy' exists and likely represents win rate, or if 'win_rate' is directly in nested 'metrics'
            if 'accuracy' in first_log_entry: 
                metric_values = [eval_logs[step].get('accuracy', 0) for step in eval_steps]
                is_win_rate = 'win_rate' in first_log_entry.get('metrics', {}) # Double check if original win_rate exists
                plot_label = 'Win Rate' if is_win_rate else 'Accuracy'
                y_label = 'Win Rate (%)' if is_win_rate else 'Accuracy (%)'
                plot_title = 'Chopped Test Set Debate Win Rate (Qwen 2.5-7B vs GPT-4)' if is_win_rate else 'Evaluation Accuracy'
                
                plt.plot(eval_steps, metric_values, color='#2ecc71', linewidth=2.0, label=plot_label)
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.title(plot_title, fontsize=24 if is_win_rate else 14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                # Save win rate as a separate PNG if it is win rate
                if is_win_rate:
                    plt.savefig(os.path.join(output_dir, 'win_rate.png'), bbox_inches='tight')
                plt.close()
            elif 'metrics' in first_log_entry and 'accuracy' in first_log_entry['metrics']:
                 # Fallback if 'accuracy' is nested under 'metrics' (less likely based on main.py)
                metric_values = [eval_logs[step].get('metrics', {}).get('accuracy', 0) for step in eval_steps]
                plt.plot(eval_steps, metric_values, color='#2ecc71', linewidth=2.0, label='Accuracy')
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Evaluation Accuracy', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()
            else:
                 print("Warning: Could not find 'accuracy' or 'win_rate' key in evaluation logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from logs directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing training logs')
    args = parser.parse_args()
    plot_metrics(args.log_dir)
