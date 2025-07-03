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
    Plot evaluation accuracy over time with a Steve Jobs-inspired retro futuristic aesthetic.
    Creates a clean, elegant visualization with dark theme and strategic bright accents.
    """
    # Load evaluation logs
    eval_logs = {}
    eval_logs_dir = os.path.join(output_dir, 'eval_logs')
    
    if not os.path.exists(eval_logs_dir):
        print(f"No evaluation logs found at {eval_logs_dir}")
        return
        
    for filename in os.listdir(eval_logs_dir):
        if filename.startswith('metrics_') and filename.endswith('.json'):
            step = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(eval_logs_dir, filename), 'r') as f:
                eval_logs[step] = json.load(f)

    if not eval_logs:
        print("No evaluation metrics found")
        return

    # Set up the retro futuristic aesthetic
    plt.style.use('dark_background')
    
    # Steve Jobs + Retro Futuristic Color Palette
    bg_color = '#0a0a0a'          # Deep black
    grid_color = '#1a1a1a'        # Subtle dark grid
    accent_color = '#00d4ff'      # Bright cyan accent
    secondary_color = '#ff6b35'    # Warm orange secondary
    text_color = '#e0e0e0'        # Light gray text
    
    # Configure matplotlib for elegant typography
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.facecolor': bg_color,
        'figure.facecolor': bg_color,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.edgecolor': grid_color,
        'axes.linewidth': 0.5,
        'grid.color': grid_color,
        'grid.linewidth': 0.5,
    })
    
    # Create the main accuracy plot
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Extract accuracy data
    eval_steps = sorted(eval_logs.keys())
    accuracy_values = [eval_logs[step].get('accuracy', 0) for step in eval_steps]
    
    # Create smooth line with glow effect
    # Main bright line
    ax.plot(eval_steps, accuracy_values, 
            color=accent_color, linewidth=3, alpha=0.9, 
            label='Evaluation Accuracy', zorder=3)
    
    # Glow effect - wider, more transparent line underneath
    ax.plot(eval_steps, accuracy_values, 
            color=accent_color, linewidth=8, alpha=0.2, zorder=2)
    
    # Add baseline performance line (horizontal line from first accuracy value)
    if accuracy_values:
        baseline_acc = accuracy_values[0]
        ax.axhline(y=baseline_acc, color=secondary_color, linewidth=2, 
                  linestyle='--', alpha=0.8, label='Baseline Performance', zorder=1)
    
    # Add subtle data points
    ax.scatter(eval_steps, accuracy_values, 
               color=accent_color, s=25, alpha=0.6, zorder=4,
               edgecolors='white', linewidth=0.5)
    
    # Elegant grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean, minimal labels
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='300', labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='300', labelpad=15)
    
    # Minimal, elegant title
    ax.set_title('Frozen Qwen 7B Performance on Math500 using Qwen 1.5B Strategies', 
                fontsize=18, fontweight='200', pad=25, color=text_color)
    
    # Light up the axes - make them more prominent
    for spine in ax.spines.values():
        spine.set_color(accent_color)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.6)
    
    # Elegant legend
    legend = ax.legend(loc='lower right', frameon=True, 
                      fancybox=False, shadow=False,
                      fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor('#1a1a1a')
    legend.get_frame().set_edgecolor(grid_color)
    
    # Set y-axis to start from a reasonable minimum
    if accuracy_values:
        y_min = max(0, min(accuracy_values) - 5)
        y_max = min(100, max(accuracy_values) + 5)
        ax.set_ylim(y_min, y_max)
    
    # Tight layout for clean presentation
    plt.tight_layout()
    
    # Save as high-quality PDF and PNG
    pdf_path = os.path.join(output_dir, 'accuracy_evolution.pdf')
    png_path = os.path.join(output_dir, 'accuracy_evolution.png')
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                facecolor=bg_color, edgecolor='none')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', 
                facecolor=bg_color, edgecolor='none')
    
    print(f"Accuracy evolution plot saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from logs directory')
    parser.add_argument('--log_dir', type=str, help='Directory containing training logs')
    args = parser.parse_args()
    plot_metrics(args.log_dir)
