import numpy as np
import matplotlib.pyplot as plt
import os

def plot_downtime_ccdf(filename):
    """
    Read down times from file and plot their empirical counter-cumulative distribution function 
    (survival function) in both linear and log-log scales using dots.
    
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
    
    # Sort durations for CCDF
    sorted_durations = np.sort(durations)
    # Calculate empirical probabilities (CCDF = 1 - CDF)
    p = 1 - (np.arange(1, len(sorted_durations) + 1) / len(sorted_durations))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot with dots
    ax1.scatter(sorted_durations, p, color='blue', s=10, alpha=0.6, label='Empirical CCDF')
    ax1.set_xlabel('Down-time Duration')
    ax1.set_ylabel('P(X > x)')
    ax1.set_title('Empirical CCDF of Down-time Durations (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log-log scale plot with dots
    # Remove zero or negative values for log plotting
    mask = sorted_durations > 0
    ax2.scatter(sorted_durations[mask], p[mask], color='red', s=10, alpha=0.6, label='Empirical CCDF')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Down-time Duration (log scale)')
    ax2.set_ylabel('P(X > x)')
    ax2.set_title('Empirical CCDF of Down-time Durations (Log-log Scale)')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    
    # Add statistics to the first plot
    stats_text = (
        f'Number of down-times: {len(durations)}\n'
        f'Mean duration: {np.mean(durations):.3f}\n'
        f'Median duration: {np.median(durations):.3f}\n'
        f'Max duration: {np.max(durations):.3f}'
    )
    
    ax1.text(
        0.95, 0.95, stats_text,
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.show()

# Run the script
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Example filename - replace with your actual filename
    file_name = "down_times_n_2_m_2_layers_9_x_0.84375_end_time_20.txt"
    
    # Construct the full path using the script directory
    full_path = os.path.join(script_dir, file_name)
    
    # Check if the file exists
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
    else:
        # Plot the CCDF
        plot_downtime_ccdf(filename=full_path)
