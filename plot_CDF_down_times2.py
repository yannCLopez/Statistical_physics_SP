import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

def plot_downtime_ccdf(filename):
    """
    Read down times from file and plot their empirical counter-cumulative distribution function 
    (survival function) in both linear and log-log scales using dots.
    Additionally, restrict plots to durations > 0.1 and perform a linearity test on the log-log plot.
    
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
    
    # Filter durations greater than 0.1
    filter_threshold = 0.01
    filtered_durations = durations[durations > filter_threshold]
    
    if len(filtered_durations) == 0:
        print(f"No down-time durations greater than {filter_threshold} found in file.")
        return
    
    # Sort durations for CCDF
    sorted_durations = np.sort(filtered_durations)
    # Calculate empirical probabilities (CCDF = 1 - CDF)
    p = 1 - (np.arange(1, len(sorted_durations) + 1) / len(sorted_durations))
    
    # Log-transform the data for linearity test
    log_durations = np.log(sorted_durations)
    log_p = np.log(p)
    
    # Perform linear regression on log-log data
    slope, intercept, r_value, p_value, std_err = linregress(log_durations, log_p)
    r_squared = r_value**2
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Linear scale plot with dots
    ax1.scatter(sorted_durations, p, color='blue', s=10, alpha=0.6, label='Empirical CCDF')
    ax1.set_xlabel('Down-time Duration')
    ax1.set_ylabel('P(X > x)')
    ax1.set_title('Empirical CCDF of Down-time Durations (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log-log scale plot with dots
    ax2.scatter(sorted_durations, p, color='red', s=10, alpha=0.6, label='Empirical CCDF')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Down-time Duration (log scale)')
    ax2.set_ylabel('P(X > x)')
    ax2.set_title('Empirical CCDF of Down-time Durations (Log-log Scale)')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    
    # Plot the regression line on log-log plot
    regression_line = intercept + slope * log_durations
    ax2.plot(sorted_durations, np.exp(regression_line), color='green', linestyle='--', linewidth=2, label='Linear Fit')
    
    # Add R-squared value to the log-log plot
    stats_text = f'R-squared: {r_squared:.4f}'
    ax2.text(
        0.95, 0.05, stats_text,
        transform=ax2.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Add statistics to the linear plot
    stats_text_linear = (
        f'Number of down-times: {len(filtered_durations)}\n'
        f'Mean duration: {np.mean(filtered_durations):.3f}\n'
        f'Median duration: {np.median(filtered_durations):.3f}\n'
        f'Max duration: {np.max(filtered_durations):.3f}'
    )
    
    ax1.text(
        0.95, 0.95, stats_text_linear,
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.show()

# Run the script
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Example filename - replace with your actual filename
    file_name = "down_times_n_2_m_2_layers_9_x_0.8_end_time_20.txt"
    
    # Construct the full path using the script directory
    full_path = os.path.join(script_dir, file_name)
    
    # Check if the file exists
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
    else:
        # Plot the CCDF
        plot_downtime_ccdf(filename=full_path)
