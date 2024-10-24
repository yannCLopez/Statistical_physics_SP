# Add this at the top of your script
import os
import sys
# Add the current directory to Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd

# Now import from the local file
from poisson_simulation_vectorized_BEN import (
    SupplyNetwork, 
    PoissonSimulation, 
    analyze_root_state_changes,
    calculate_analytical_F
)

# Rest of your code remains the same...

def create_x_values() -> np.ndarray:
    """Create array of x values with dense sampling in region of interest"""
    # Regular sampling from 0.6 to 0.95
    x_coarse = np.arange(0.6, 0.79, 0.05)
    x_coarse_upper = np.arange(0.87, 0.96, 0.05)
    
    # Dense sampling from 0.79 to 0.86
    x_dense = np.arange(0.79, 0.861, 0.01)
    
    # Combine and sort all x values
    return np.sort(np.concatenate([x_coarse, x_dense, x_coarse_upper]))

def run_duration_analysis(x: float, n: int, m: int, num_layers: int, 
                         end_time: float, num_trials: int) -> Tuple[float, float]:
    """
    Run multiple trials for a given x value and return average durations
    """
    up_durations = []
    down_durations = []
    
    for trial in range(num_trials):
        # Create network and run simulation
        network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
        sim = PoissonSimulation(network)
        root_state_changes = sim.run_until(end_time)
        
        # Analyze results
        # to-do: change path
        #c
        stats = analyze_root_state_changes(root_state_changes, end_time, 
                                         output_file=f"temp_down_times_x{x}_trial{trial}.txt")
        
        if stats['average_operational_period'] > 0:
            up_durations.append(stats['average_operational_period'])
        if stats['average_non_operational_period'] > 0:
            down_durations.append(stats['average_non_operational_period'])
    
    # Calculate averages across trials
    avg_up = np.mean(up_durations) if up_durations else 0
    avg_down = np.mean(down_durations) if down_durations else 0
    
    return avg_up, avg_down

def plot_duration_analysis(x_values: np.ndarray, up_times: List[float], 
                         down_times: List[float], save_path: str = None,
                         x_crit: float = None):
    """Create and save plots of duration analysis with critical point"""
    # Plot both durations on same figure with different y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot uptime durations
    ax1.plot(x_values, up_times, 'b-', label='Uptime Duration')
    ax1.set_xlabel('x (Edge Operational Probability)')
    ax1.set_ylabel('Average Uptime Duration', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for downtime durations
    #c
    ax2 = ax1.twinx()
    ax2.plot(x_values, down_times, 'r-', label='Downtime Duration')
    ax2.set_ylabel('Average Downtime Duration', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add vertical line at x_crit
    if x_crit is not None:
        # Plot vertical line
        ax1.axvline(x=x_crit, color='k', linestyle='--', alpha=0.7)
        
        # Add label for x_crit
        # Position it slightly above the x-axis and centered on the vertical line
        ax1.text(x_crit, ax1.get_ylim()[0], 
                f'x_c = {x_crit:.3f}',
                horizontalalignment='center',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3))
    
    # Add title and grid
    plt.title('Average Up/Down Time Durations vs Edge Operational Probability')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Highlight the dense sampling region
    plt.axvspan(0.79, 0.86, color='gray', alpha=0.1, label='Dense Sampling Region')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set simulation parameters
    n = 2
    m = 2
    x_crit = 0.843750000000043
    num_layers = 6
    end_time = 1000
    num_trials = 2  # Number of trials to average over for each x value
    
    # Create x values
    x_values = create_x_values()
    
    # Initialize lists to store results
    up_times = []
    down_times = []
    
    # Run analysis for each x value
    total_runs = len(x_values)
    for i, x in enumerate(x_values):
        print(f"\nAnalyzing x = {x:.3f} ({i+1}/{total_runs})")
        avg_up, avg_down = run_duration_analysis(x, n, m, num_layers, end_time, num_trials)
        up_times.append(avg_up)
        down_times.append(avg_down)
        
        # Print intermediate results
        print(f"Average uptime duration: {avg_up:.3f}")
        print(f"Average downtime duration: {avg_down:.3f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'x': x_values,
        'avg_uptime': up_times,
        'avg_downtime': down_times
    })

    # Save results to CSV
    save_filename_data = f'duration_analysis_results_n_{n}_m_{m}_layers_{num_layers}_endtime_{end_time}_numtrials_{num_trials}.csv'
    save_path_data = os.path.join(script_dir, save_filename_data)
    results_df.to_csv(save_path_data, index=False)
    
    # Create and save plot
    # Create save path in the same directory as the script
    save_filename_plot = f'duration_analysis_plot_n_{n}_m_{m}_layers_{num_layers}_endtime_{end_time}.png'

    save_path_plot = os.path.join(script_dir, save_filename_plot)
    
    # Create and save plot
    plot_duration_analysis(x_values, up_times, down_times, save_path_plot, x_crit=x_crit)

if __name__ == "__main__":
    main()


