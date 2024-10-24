import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

def plot_downtime_distributions(filename):
    """
    Read down times from file and plot their empirical CCDF and CDF in various scales.
    CCDF is plotted in linear and log-log scales.
    CDF is plotted in linear and log-log scales.

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

    # ============================
    # Plotting CCDF Figure
    # ============================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Linear scale CCDF plot
    ax1.scatter(sorted_durations, ccdf_p, color='blue', s=10, alpha=0.6, label='Empirical CCDF')
    ax1.set_xlabel('Down-time Duration')
    ax1.set_ylabel('P(X > x)')
    ax1.set_title('Empirical CCDF of Down-time Durations (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Log-log scale CCDF plot
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

    # Add statistics to the linear CCDF plot
    stats_text_ccdf_linear = (
        f'Number of down-times: {n}\n'
        f'Mean duration: {np.mean(filtered_durations):.3f}\n'
        f'Median duration: {np.median(filtered_durations):.3f}\n'
        f'Max duration: {np.max(filtered_durations):.3f}'
    )

    ax1.text(
        0.95, 0.95, stats_text_ccdf_linear,
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking to allow the next figure to pop up

    # ============================
    # Plotting CDF Figure
    # ============================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))

    # Linear scale CDF plot
    ax3.scatter(sorted_durations, cdf_p, color='purple', s=10, alpha=0.6, label='Empirical CDF')
    ax3.set_xlabel('Down-time Duration')
    ax3.set_ylabel('P(X ≤ x)')
    ax3.set_title('Empirical CDF of Down-time Durations (Linear Scale)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Log-log scale CDF plot
    ax4.scatter(sorted_durations, cdf_p, color='orange', s=10, alpha=0.6, label='Empirical CDF')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Down-time Duration (log scale)')
    ax4.set_ylabel('P(X ≤ x) (log scale)')
    ax4.set_title('Empirical CDF of Down-time Durations (Log-log Scale)')
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    ax4.legend()

    # Handle potential -inf in log-log CDF plot by excluding zero or near-zero CDF values
    # Since we filtered durations > 0.01, cdf_p should start from 1/n > 0
    # However, to ensure numerical stability, we can slightly adjust CDF values if needed
    # Here, we proceed directly

    # Add statistics to the linear CDF plot
    stats_text_cdf_linear = (
        f'Number of down-times: {n}\n'
        f'Mean duration: {np.mean(filtered_durations):.3f}\n'
        f'Median duration: {np.median(filtered_durations):.3f}\n'
        f'Max duration: {np.max(filtered_durations):.3f}'
    )

    ax3.text(
        0.95, 0.95, stats_text_cdf_linear,
        transform=ax3.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Optional: Add linear fit or other statistics to log-log CDF plot
    # Here, we skip it for simplicity

    plt.tight_layout()
    plt.show()

# Run the script
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example filename - replace with your actual filename
    file_name = "down_times_n_2_m_2_layers_9_x_0.8_end_time_200.txt"

    # Construct the full path using the script directory
    full_path = os.path.join(script_dir, file_name)

    # Check if the file exists
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
    else:
        # Plot the CCDF and CDF
        plot_downtime_distributions(filename=full_path)
