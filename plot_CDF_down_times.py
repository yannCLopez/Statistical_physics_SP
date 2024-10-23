import numpy as np
import matplotlib.pyplot as plt

def plot_downtime_cdf(filename):
    """
    Read down times from file and plot their empirical CDF in both linear and log-log scales
    
    Parameters:
    -----------
    filename : str
        Name of file containing down-time durations (one per line, with header)
    """
    # Read durations from file
    with open(filename, 'r') as f:
        # Skip header
        next(f)
        # Read durations into numpy array
        durations = np.array([float(line.strip()) for line in f])
    
    if len(durations) == 0:
        print("No down-time durations found in file")
        return
    
    # Sort durations for CDF
    sorted_durations = np.sort(durations)
    # Calculate empirical probabilities
    p = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot
    ax1.step(sorted_durations, p, where='post', label='Empirical CDF')
    ax1.set_xlabel('Down-time Duration')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('Empirical CDF of Down-time Durations (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale plot
    # Remove zero values for log plotting
    mask = sorted_durations > 0
    ax2.step(sorted_durations[mask], p[mask], where='post', label='Empirical CDF')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Down-time Duration (log scale)')
    ax2.set_ylabel('Cumulative Probability (log scale)')
    ax2.set_title('Empirical CDF of Down-time Durations (Log-log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to the first plot
    stats_text = f'Number of down-times: {len(durations)}\n'
    stats_text += f'Mean duration: {np.mean(durations):.3f}\n'
    stats_text += f'Median duration: {np.median(durations):.3f}\n'
    stats_text += f'Max duration: {np.max(durations):.3f}'
    
    ax1.text(0.95, 0.05, stats_text,
             transform=ax1.transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Run the script
if __name__ == "__main__":
    file_folder = "/Users/yanncalvolopez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Career/RA Ben/Statistical physics of supply chains/down_times_stats"
    file_name = "down_times_n_2_m_2_layers_5_x_0.843750000000043_end_time_1000.txt"
    plot_downtime_cdf(filename=f"{file_folder}/{file_name}")