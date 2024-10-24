import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

def load_duration_data(filepath: str) -> pd.DataFrame:
    """
    Load duration analysis data from CSV file.
    
    Args:
        filepath: Path to the CSV file containing duration analysis results
        
    Returns:
        DataFrame containing the analysis results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    df = pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        required_columns = ['x', 'avg_uptime', 'avg_downtime']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
        
    return df

def filter_x_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Allow user to filter out specific x values from the dataset.
    
    Args:
        data: DataFrame containing the analysis results
        
    Returns:
        Filtered DataFrame
    """
    print("\nCurrent x values in dataset:")
    for i, x in enumerate(data['x'].values, 1):
        print(f"{i}. x = {x:.3f}")
    
    filtered_data = data.copy()
    
    while True:
        choice = input("\nEnter numbers of x values to exclude (comma-separated) or press Enter to keep all: ")
        
        if not choice.strip():
            break
            
        try:
            # Convert input to list of indices
            indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            
            # Validate indices
            if all(0 <= idx < len(data) for idx in indices):
                # Create mask for filtering
                mask = ~filtered_data.index.isin(indices)
                filtered_data = filtered_data[mask]
                
                print("\nUpdated x values:")
                for i, x in enumerate(filtered_data['x'].values, 1):
                    print(f"{i}. x = {x:.3f}")
                
                if input("\nFilter more x values? (y/n): ").lower() != 'y':
                    break
            else:
                print("Invalid indices. Please try again.")
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers.")
    
    return filtered_data

def create_duration_plot(
    data: pd.DataFrame,
    x_crit: Optional[float] = None,
    title: Optional[str] = None,
    dense_region: tuple = (0.79, 0.86),
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Create duration analysis plot with dual y-axes.
    
    Args:
        data: DataFrame containing x, avg_uptime, and avg_downtime columns
        x_crit: Critical x value to mark with vertical line
        title: Custom title for the plot
        dense_region: Tuple of (start, end) x values for dense sampling region
        figsize: Tuple of (width, height) for the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot uptime durations
    line1 = ax1.plot(data['x'], data['avg_uptime'], 'b-', label='Uptime Duration')
    ax1.set_xlabel('x (Edge Operational Probability)')
    ax1.set_ylabel('Average Uptime Duration', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for downtime durations
    ax2 = ax1.twinx()
    line2 = ax2.plot(data['x'], data['avg_downtime'], 'r-', label='Downtime Duration')
    ax2.set_ylabel('Average Downtime Duration', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add critical point line if provided
    if x_crit is not None:
        ax1.axvline(x=x_crit, color='k', linestyle='--', alpha=0.7)
        ax1.text(
            x_crit, 
            ax1.get_ylim()[0],
            f'x_c = {x_crit:.3f}',
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3)
        )
    
    # Add title
    if title is None:
        title = 'Average Up/Down Time Durations vs Edge Operational Probability'
    plt.title(title)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    # Highlight dense sampling region
    plt.axvspan(dense_region[0], dense_region[1], color='gray', alpha=0.1, 
                label='Dense Sampling Region')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def extract_params_from_filename(filename: str) -> dict:
    """
    Extract simulation parameters from filename.
    
    Args:
        filename: Name of the CSV file containing duration analysis results
        
    Returns:
        Dictionary containing extracted parameters
    """
    params = {}
    try:
        # Split filename by underscores and remove extension
        parts = filename.replace('.csv', '').split('_')
        
        # Extract parameters
        for i, part in enumerate(parts):
            if part == 'n':
                params['n'] = int(parts[i + 1])
            elif part == 'm':
                params['m'] = int(parts[i + 1])
            elif part == 'layers':
                params['num_layers'] = int(parts[i + 1])
            elif part == 'endtime':
                params['end_time'] = int(parts[i + 1])
            elif part == 'numtrials':
                params['num_trials'] = int(parts[i + 1])
    except Exception as e:
        print(f"Warning: Could not extract all parameters from filename: {e}")
    
    return params

def select_csv_file(directory: str) -> Optional[str]:
    """
    Present available CSV files to user and let them select one.
    
    Args:
        directory: Directory to search for CSV files
        
    Returns:
        Selected filename or None if no selection made
    """
    # Find CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') 
                 and 'duration_analysis_results' in f]
    
    if not csv_files:
        print("No duration analysis CSV files found in the directory.")
        return None
    
    # Print available files
    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter the number of the file to process (or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(csv_files):
                return csv_files[index]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parameters
    x_crit = 0.843750000000043
    
    # Let user select CSV file
    selected_file = select_csv_file(script_dir)
    
    if selected_file is None:
        print("No file selected. Exiting.")
        return
    
    try:
        # Load data
        filepath = os.path.join(script_dir, selected_file)
        data = load_duration_data(filepath)
        
        # Allow user to filter x values
        filtered_data = filter_x_values(data)
        
        if len(filtered_data) == 0:
            print("All data points were filtered out. Exiting.")
            return
        
        # Extract parameters from filename
        params = extract_params_from_filename(selected_file)
        
        # Create title using parameters
        title = f"Duration Analysis (n={params.get('n', '?')}, m={params.get('m', '?')}, "
        title += f"layers={params.get('num_layers', '?')}, "
        title += f"trials={params.get('num_trials', '?')})"
        
        # Create plot
        fig = create_duration_plot(filtered_data, x_crit=x_crit, title=title)
        
        # Save plot
        plot_filename = selected_file.replace('.csv', '_filtered_plot.png')
        save_path = os.path.join(script_dir, plot_filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"\nCreated plot: {plot_filename}")
        
    except Exception as e:
        print(f"Error processing {selected_file}: {e}")

if __name__ == "__main__":
    main()