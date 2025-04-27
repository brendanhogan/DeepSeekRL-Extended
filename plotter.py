import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.backends.backend_pdf import PdfPages

# Define Art Deco / Retro-Futuristic Style elements
# Palette: Gold, Silver/Gray, Deep Blue, Teal, Orange/Red contrast
RTF_AD_COLORS = ['#FFD700', '#C0C0C0', '#00008B', '#20B2AA', '#FF4500', '#4682B4', '#DAA520', '#5F9EA0']
RTF_AD_STYLE = {
    "axes.facecolor": "#2E2E2E",      # Dark background
    "axes.edgecolor": "#C0C0C0",      # Silver edges
    "axes.grid": True,
    "grid.color": "#777777",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.labelcolor": "#FFFFFF",
    "axes.titlecolor": "#FFD700",    # Gold title
    "xtick.color": "#C0C0C0",
    "ytick.color": "#C0C0C0",
    "legend.facecolor": "#3E3E3E",
    "legend.edgecolor": "#C0C0C0",
    "legend.title_fontsize": "medium",
    "figure.facecolor": "#1E1E1E",    # Very dark figure background
    "figure.edgecolor": "#1E1E1E",
    "font.family": "sans-serif",      # Use a clean sans-serif font
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], # Common sans-serif fallbacks
    "axes.titlesize": 16, 
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "lines.markersize": 6
}
plt.rcParams.update(RTF_AD_STYLE)

def moving_average(data, window_size=5):
    """Calculate moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_metrics(output_dir):
    """
    Plot training metrics from training_logs directory and evaluation results.
    Creates a PDF with plots and a separate PNG for win rate.
    Uses a modern, professional style with custom color palette.
    """
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    # Load training logs
    train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
    try:
        with open(train_logs_path, 'r') as f:
            train_logs = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Training log file not found at {train_logs_path}. Skipping training plots.")
        train_logs = None

    eval_results = {} # Initialize BEFORE checking directory
    # Load individual evaluation summary files
    eval_logs_dir = os.path.join(output_dir, 'eval_logs')
    if not os.path.isdir(eval_logs_dir):
        print(f"Warning: Evaluation logs directory not found at {eval_logs_dir}. Skipping evaluation plots.")
    else:
        for filename in sorted(os.listdir(eval_logs_dir)):
            if filename.startswith('summary_metrics_') and filename.endswith('.json'):
                try:
                    step_str = filename.split('_')[-1].split('.')[0]
                    if step_str.isdigit(): # Process only numeric steps
                        step = int(step_str)
                        filepath = os.path.join(eval_logs_dir, filename)
                        with open(filepath, 'r') as f:
                            eval_data = json.load(f)
                            # Store the relevant part (win_rate and metrics dict)
                            if 'win_rate' in eval_data: # Ensure win_rate exists
                                eval_results[step] = {
                                    'win_rate': eval_data['win_rate'],
                                    'metrics': eval_data.get('metrics', {}) # Include metrics if present
                                }
                    else:
                        print(f"Skipping non-numeric eval log file: {filename}")
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not process evaluation file {filename}: {e}")

    # Set style and color palette
    colors = RTF_AD_COLORS
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    win_rate_png_path = os.path.join(output_dir, 'eval_win_rate.png') # Define path for separate PNG

    with PdfPages(pdf_path) as pdf:
        
        # Plot training metrics if logs exist
        if train_logs:
            # Determine which reward metrics exist in the logs
            try:
                sample_metrics = next(iter(train_logs.values()))
                reward_metrics = [key for key in sample_metrics.keys() if key.startswith('rewards/')]
            except StopIteration: # Handle empty train_logs
                reward_metrics = []
                sample_metrics = {}

            # Assign colors for reward metrics
            reward_colors = {metric: colors[i % len(colors)] for i, metric in enumerate(reward_metrics)}

            # Plot reward metrics
            for metric in reward_metrics:
                plt.figure(figsize=(12,7))
                color = reward_colors[metric]
                steps = [int(x) for x in train_logs.keys()]
                values = [metrics.get(metric, 0) for metrics in train_logs.values()]
                
                # Plot raw data with low alpha
                plt.plot(steps, values, color=color, alpha=0.4, linewidth=1.0, linestyle=':', label='Raw data')
                
                # Calculate and plot moving average if we have enough data points
                if len(values) > 5:
                    ma_values = moving_average(values)
                    ma_steps = steps[len(steps)-len(ma_values):]
                    plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, marker='.', label='Moving average')
                
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(f'{metric.split("/")[-1].replace("_", " ").title()}', fontsize=12)
                plt.title(f'{metric.split("/")[-1].replace("_", " ").title()}', pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

            # Plot other metrics that exist in both evaluators
            common_metrics = ['learning_rate', 'reward_std', 'loss', 'kl']
            metric_colors = {
                'learning_rate': colors[1 % len(colors)],
                'reward_std': colors[2 % len(colors)],
                'loss': colors[3 % len(colors)],
                'kl': colors[4 % len(colors)],
            }
            
            for metric in common_metrics:
                if any(metric in metrics for metrics in train_logs.values()):
                    plt.figure(figsize=(12,7))
                    steps = [int(x) for x in train_logs.keys()]
                    values = [metrics.get(metric, np.nan) for metrics in train_logs.values()] # Use NaN for missing

                    # Plot raw data with low alpha
                    plt.plot(steps, values, color=metric_colors[metric], alpha=0.4, linewidth=1.0, linestyle=':', label=f'{metric} (Raw)')
                    if len(values) > 5:
                        ma_values = moving_average(values)
                        ma_steps = steps[len(steps)-len(ma_values):]
                        plt.plot(ma_steps, ma_values, color=metric_colors[metric], linewidth=2.5, marker='.', label=f'{metric} (MA)')

                    plt.xlabel('Training Steps', fontsize=12)
                    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                    plt.title(f'{metric.replace("_", " ").title()}', pad=20)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

        # Plot evaluation metrics if eval_results exist
        if eval_results:
            eval_steps = sorted(eval_results.keys())
            if not eval_steps: # Skip if no numeric steps found
                print("No evaluation steps found to plot win rate.")
            else:
                # Plot win rate
                plt.figure(figsize=(12,7))
                # Extract win rates - handle potential missing keys gracefully
                win_rates = [eval_results[step].get('win_rate', None) for step in eval_steps]
                # Filter out None values if any steps missed saving a win rate
                valid_steps = [step for step, rate in zip(eval_steps, win_rates) if rate is not None]
                valid_win_rates = [rate for rate in win_rates if rate is not None]

                if valid_steps:
                    win_rate_color = colors[0 % len(colors)] # Use first color for win rate
                    plt.plot(valid_steps, valid_win_rates, color=win_rate_color, linewidth=2.5, marker='D', linestyle='-', label='Win Rate') # Diamond marker
                    plt.xlabel('Training Steps', fontsize=12)
                    plt.ylabel('Evaluation Win Rate (%)', fontsize=12)
                    plt.title('Qwen2.5-7B vs. gpt-4o-mini Debate Win Rate', pad=20) # Dynamic title
                    plt.legend()
                    pdf.savefig(bbox_inches='tight') # Save to combined PDF
                    plt.savefig(win_rate_png_path, bbox_inches='tight') # Save separate PNG
                    plt.close()
                else:
                    print("No valid win rate data found to plot.")

    print(f"Training plots saved to {pdf_path}")
    if os.path.exists(win_rate_png_path):
        print(f"Evaluation win rate plot saved to {win_rate_png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from logs directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing training logs')
    args = parser.parse_args()
    plot_metrics(args.log_dir)
