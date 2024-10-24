import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

def plot_downtime_distributions(filename):
    """
    Read down times from file and plot their empirical CCDF and CDF in various scales.
    CCDF is plotted in linear and log-log scales.
    CDF is plotted in linear and log-log scales.
    Saves all plots in a single figure file with '_plot' appended to the original filename.

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

    # Filter durations greater than the threshold
    filter_threshold = 0.01
    filtered_durations = durations[durations > filter_threshold]

    if len(filtered_durations) == 0:
        print(f"No down-time durations greater than {filter_threshold} found in file.")
        return

    # Sort durations for CCDF and CDF
    sorted_durations = np.sort(filtered_durations)
    n = len(sorted_durations)

    # Calculate empirical CCDF: P(X > x)
    ccdf_p = 1 - (np.arange(1, n + 1) / n)

    # Calculate empirical CDF: P(X ≤ x)
    cdf_p = np.arange(1, n + 1) / n

    # Log-transform the data for linearity test on CCDF
    log_durations = np.log(sorted_durations)
    log_ccdf_p = np.log(ccdf_p)

    # Perform linear regression on log-log CCDF data
    slope, intercept, r_value, p_value, std_err = linregress(log_durations, log_ccdf_p)
    r_squared = r_value**2

    # Create a single figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 16))

    # ============================
    # CCDF Plots (Top Row)
    # ============================
    # Linear scale CCDF plot
    ax1 = fig.add_subplot(221)
    ax1.scatter(sorted_durations, ccdf_p, color='blue', s=10, alpha=0.6, label='Empirical CCDF')
    ax1.set_xlabel('Down-time Duration')
    ax1.set_ylabel('P(X > x)')
    ax1.set_title('Empirical CCDF of Down-time Durations (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Log-log scale CCDF plot
    ax2 = fig.add_subplot(222)
    ax2.scatter(sorted_durations, ccdf_p, color='red', s=10, alpha=0.6, label='Empirical CCDF')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Down-time Duration (log scale)')
    ax2.set_ylabel('P(X > x)')
    ax2.set_title('Empirical CCDF of Down-time Durations (Log-log Scale)')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()

    # Plot the regression line on log-log CCDF plot
    regression_line = intercept + slope * log_durations
    ax2.plot(sorted_durations, np.exp(regression_line), color='green', linestyle='--', linewidth=2, label='Linear Fit')

    # ============================
    # CDF Plots (Bottom Row)
    # ============================
    # Linear scale CDF plot
    ax3 = fig.add_subplot(223)
    ax3.scatter(sorted_durations, cdf_p, color='purple', s=10, alpha=0.6, label='Empirical CDF')
    ax3.set_xlabel('Down-time Duration')
    ax3.set_ylabel('P(X ≤ x)')
    ax3.set_title('Empirical CDF of Down-time Durations (Linear Scale)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Log-log scale CDF plot
    ax4 = fig.add_subplot(224)
    ax4.scatter(sorted_durations, cdf_p, color='orange', s=10, alpha=0.6, label='Empirical CDF')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Down-time Duration (log scale)')
    ax4.set_ylabel('P(X ≤ x) (log scale)')
    ax4.set_title('Empirical CDF of Down-time Durations (Log-log Scale)')
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    ax4.legend()

    # Add statistics to plots
    stats_text = (
        f'Number of down-times: {n}\n'
        f'Mean duration: {np.mean(filtered_durations):.3f}\n'
        f'Median duration: {np.median(filtered_durations):.3f}\n'
        f'Max duration: {np.max(filtered_durations):.3f}'
    )

    # Add R-squared value to the log-log CCDF plot
    stats_text_ccdf = f'R-squared: {r_squared:.4f}'
    ax2.text(
        0.95, 0.05, stats_text_ccdf,
        transform=ax2.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Add statistics to other plots
    for ax in [ax1, ax3]:
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()

    # Generate output filename by appending '_plot' before the extension
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_plot{ext}"
    if ext.lower() != '.png':
        output_filename = f"{base}_plot.png"

    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Plot saved as: {output_filename}")

# Run the script
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example filename - replace with your actual filename
    file_name = "down_times_n_2m3layers6x0.8770978009end_time1000.txt"

    # Construct the full path using the script directory
    full_path = os.path.join(script_dir, file_name)

    # Check if the file exists
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
    else:
        # Plot the CCDF and CDF
        plot_downtime_distributions(filename=full_path)
