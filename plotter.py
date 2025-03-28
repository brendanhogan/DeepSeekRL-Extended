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
    with open(train_logs_path, 'r') as f:
        train_logs = json.load(f)

    # Load evaluation logs
    eval_logs = {}
    eval_logs_dir = os.path.join(output_dir, 'eval_logs')
    for filename in os.listdir(eval_logs_dir):
        if filename.startswith('metrics_') and filename.endswith('.json'):
            step = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(eval_logs_dir, filename), 'r') as f:
                eval_logs[step] = json.load(f)

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
            
            # Plot win rate for debate or accuracy for GSM8k
            plt.figure(figsize=(12,7))
            if 'win_rate' in eval_logs[eval_steps[0]]['metrics']:
                metric_values = [eval_logs[step]['metrics']['win_rate'] for step in eval_steps]
                plt.plot(eval_steps, metric_values, color='#2ecc71', linewidth=2.0, label='Win Rate')
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel('Win Rate (%)', fontsize=12)
                plt.title('Test Set Debate Win Rate (Qwen 2.5-1.5B vs GPT-4)', fontsize=24, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.savefig(os.path.join(output_dir, 'win_rate.png'), bbox_inches='tight')
                plt.close()
            else:
                metric_values = [eval_logs[step].get('accuracy', 0) for step in eval_steps]
                plt.plot(eval_steps, metric_values, color='#2ecc71', linewidth=2.0, label='Accuracy')
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.title('Evaluation Accuracy', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from logs directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing training logs')
    args = parser.parse_args()
    plot_metrics(args.log_dir)
