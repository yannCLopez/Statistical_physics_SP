import numpy as np
from itertools import combinations
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
from poisson_simulation_vectorized_BEN import SupplyNetwork, PoissonSimulation

class NodeCorrelationAnalyzer:
    def __init__(self, network: SupplyNetwork, num_samples: int = 100):
        self.network = network
        self.num_samples = num_samples
        self.node_states_history = []
        self.sample_nodes = set()
        self.time_points = []
        
    def get_all_nodes(self) -> Set[int]:
        """Get all node IDs in the network"""
        nodes = {0}  # Start with root
        for parent in self.network.children:
            for type_children in self.network.children[parent].values():
                nodes.update(type_children)
        return nodes
        
    def get_nodes_by_tier(self) -> Dict[int, Set[int]]:
        """Group nodes by their tier (distance from root)"""
        tiers = defaultdict(set)
        tiers[0].add(0)  # Root is tier 0
        
        # BFS to assign tiers
        queue = [(0, 0)]  # (node, tier)
        seen = {0}
        
        while queue:
            node, tier = queue.pop(0)
            for type_children in self.network.children[node].values():
                for child in type_children:
                    if child not in seen:
                        tiers[tier + 1].add(child)
                        queue.append((child, tier + 1))
                        seen.add(child)
                        
        return dict(tiers)
    
    def sample_nodes_random(self) -> Set[int]:
        """Randomly sample nodes from the network"""
        all_nodes = list(self.get_all_nodes())
        num_samples = min(self.num_samples, len(all_nodes))
        return set(random.sample(all_nodes, num_samples))
    
    def sample_nodes_stratified(self) -> Set[int]:
        """Sample nodes with stratification by tier"""
        tiers = self.get_nodes_by_tier()
        samples_per_tier = max(1, self.num_samples // len(tiers))
        stratified_samples = set()
        
        for tier_nodes in tiers.values():
            num_tier_samples = min(samples_per_tier, len(tier_nodes))
            stratified_samples.update(random.sample(list(tier_nodes), num_tier_samples))
            
        # If we need more samples to reach num_samples, randomly sample from all remaining nodes
        remaining_samples = self.num_samples - len(stratified_samples)
        if remaining_samples > 0:
            remaining_nodes = list(self.get_all_nodes() - stratified_samples)
            if remaining_nodes:
                additional_samples = random.sample(remaining_nodes, 
                                                min(remaining_samples, len(remaining_nodes)))
                stratified_samples.update(additional_samples)
                
        return stratified_samples
    
    def record_node_states(self, time_point: float):
        """Record the states of sampled nodes at a given time point"""
        states = {node: self.network.node_states[node] for node in self.sample_nodes}
        self.node_states_history.append(states)
        self.time_points.append(time_point)
    
    def calculate_correlations(self) -> Tuple[float, List[float]]:
        """Calculate average correlation between pairs of sampled nodes"""
        # Convert node states to time series (1 for operational, 0 for non-operational)
        node_series = {}
        for node in self.sample_nodes:
            series = [1 if states[node] else 0 for states in self.node_states_history]
            node_series[node] = series
            
        # Calculate correlations between all pairs
        correlations = []
        for node1, node2 in combinations(self.sample_nodes, 2):
            series1 = node_series[node1]
            series2 = node_series[node2]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(series1, series2)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)
                
        avg_correlation = np.mean(correlations) if correlations else 0
        return avg_correlation, correlations
    
    def run_analysis(self, end_time: float, sampling_interval: float, 
                    stratified: bool = False) -> Tuple[float, List[float]]:
        """Run full correlation analysis"""
        # Sample nodes
        self.sample_nodes = (self.sample_nodes_stratified() if stratified 
                           else self.sample_nodes_random())
        
        # Clear previous history
        self.node_states_history = []
        self.time_points = []
        
        # Run simulation and record states
        sim = PoissonSimulation(self.network)
        current_time = 0
        
        while current_time < end_time:
            self.record_node_states(current_time)
            current_time += sampling_interval
            
            # Run simulation until next sampling point
            state_changes = sim.run_until(current_time)
            
        return self.calculate_correlations()

def run_correlation_study(n: int, m: int, num_layers: int, x: float, 
                         num_samples: int = 100, end_time: float = 10.0,
                         sampling_interval: float = 0.1) -> dict:
    """Run both random and stratified correlation analysis"""
    # Create network
    network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
    analyzer = NodeCorrelationAnalyzer(network, num_samples)
    
    # Run random sampling analysis
    random_avg, random_corrs = analyzer.run_analysis(
        end_time, sampling_interval, stratified=False)
    
    # Run stratified sampling analysis
    strat_avg, strat_corrs = analyzer.run_analysis(
        end_time, sampling_interval, stratified=True)
    
    return {
        'random_average': random_avg,
        'random_correlations': random_corrs,
        'stratified_average': strat_avg,
        'stratified_correlations': strat_corrs
    }

def plot_correlation_distributions(results: dict, save_path: str = None):
    """Plot distributions of correlations for both sampling methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histograms
    n_bins = 30
    random_hist = ax1.hist(results['random_correlations'], bins=n_bins, alpha=0.7, 
                          label='Random Sampling', density=True, color='blue')
    strat_hist = ax2.hist(results['stratified_correlations'], bins=n_bins, alpha=0.7, 
                         label='Stratified Sampling', density=True, color='orange')
    
    # Add mean and std lines
    for ax, data, color, title in [
        (ax1, results['random_correlations'], 'blue', 'Random Sampling'),
        (ax2, results['stratified_correlations'], 'orange', 'Stratified Sampling')
    ]:
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(mean, color=color, linestyle='--', 
                   label=f'Mean: {mean:.3f}\nStd: {std:.3f}')
        ax.axvline(mean + std, color=color, linestyle=':', alpha=0.5)
        ax.axvline(mean - std, color=color, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Pairwise Correlation Coefficient')
        ax.set_ylabel('Density')
        ax.set_title(f'{title}\n({len(data)} pairs)')
        ax.legend()
        
        # Add text with pair count
        ax.text(0.02, 0.98, f'Number of pairs: {len(data):,}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle('Distribution of Pairwise Node State Correlations', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sampling_distribution(network: SupplyNetwork, sampled_nodes: Set[int]) -> Dict:
    """Analyze how the samples are distributed across tiers"""
    analyzer = NodeCorrelationAnalyzer(network, len(sampled_nodes))
    tiers = analyzer.get_nodes_by_tier()
    
    # Count samples per tier
    tier_distribution = defaultdict(int)
    tier_totals = defaultdict(int)
    
    # Find which tier each node belongs to
    for tier, nodes in tiers.items():
        tier_totals[tier] = len(nodes)
        for node in nodes:
            if node in sampled_nodes:
                tier_distribution[tier] += 1
    
    return {
        'distribution': dict(tier_distribution),
        'totals': dict(tier_totals),
        'percentages': {tier: (count/tier_totals[tier])*100 
                       for tier, count in tier_distribution.items() 
                       if tier_totals[tier] > 0}
    }

def create_x_values() -> np.ndarray:
    """Create array of x values with dense sampling in region of interest"""
    # Regular sampling from 0.6 to 0.95
    x_coarse = np.arange(0.6, 0.79, 0.05)
    x_coarse_upper = np.arange(0.87, 0.96, 0.05)
    
    # Dense sampling from 0.79 to 0.86 (around critical point)
    x_dense = np.arange(0.79, 0.861, 0.01)
    
    # Combine and sort all x values
    return np.sort(np.concatenate([x_coarse, x_dense, x_coarse_upper]))

def run_correlation_sweep(n: int, m: int, num_layers: int, x_values: np.ndarray,
                         num_samples: int = 100, end_time: float = 10.0,
                         sampling_interval: float = 0.1) -> dict:
    """Run correlation analysis for multiple x values"""
    random_avgs = []
    random_stds = []
    strat_avgs = []
    strat_stds = []
    
    total_runs = len(x_values)
    
    for i, x in enumerate(x_values):
        print(f"\nAnalyzing x = {x:.3f} ({i+1}/{total_runs})")
        
        # Run analysis for this x value
        results = run_correlation_study(
            n=n, m=m, num_layers=num_layers, x=x,
            num_samples=num_samples, end_time=end_time,
            sampling_interval=sampling_interval
        )
        
        # Store results
        random_avgs.append(results['random_average'])
        random_stds.append(np.std(results['random_correlations']))
        strat_avgs.append(results['stratified_average'])
        strat_stds.append(np.std(results['stratified_correlations']))
        
        # Print intermediate results
        print(f"Random sampling average correlation: {results['random_average']:.3f}")
        print(f"Stratified sampling average correlation: {results['stratified_average']:.3f}")
    
    return {
        'x_values': x_values,
        'random_averages': np.array(random_avgs),
        'random_stds': np.array(random_stds),
        'stratified_averages': np.array(strat_avgs),
        'stratified_stds': np.array(strat_stds)
    }

def plot_correlation_sweep(results: dict, x_crit: float = None, save_path: str = None):
    """Plot how correlations vary with x for both sampling methods"""
    plt.figure(figsize=(12, 8))
    
    x_values = results['x_values']
    
    # Plot average correlations with standard deviation bands
    plt.plot(x_values, results['random_averages'], 'b-', label='Random Sampling', linewidth=2)
    plt.fill_between(x_values, 
                     results['random_averages'] - results['random_stds'],
                     results['random_averages'] + results['random_stds'],
                     color='blue', alpha=0.2)
    
    plt.plot(x_values, results['stratified_averages'], 'r-', label='Stratified Sampling', linewidth=2)
    plt.fill_between(x_values, 
                     results['stratified_averages'] - results['stratified_stds'],
                     results['stratified_averages'] + results['stratified_stds'],
                     color='red', alpha=0.2)
    
    # Add vertical line at x_crit if provided
    if x_crit is not None:
        plt.axvline(x=x_crit, color='k', linestyle='--', alpha=0.7)
        plt.text(x_crit, plt.ylim()[0], f'x_c = {x_crit:.3f}',
                horizontalalignment='center', verticalalignment='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Highlight dense sampling region
    plt.axvspan(0.79, 0.86, color='gray', alpha=0.1, label='Dense Sampling Region')
    
    plt.xlabel('x (Edge Operational Probability)')
    plt.ylabel('Average Correlation Coefficient')
    plt.title('Node State Correlations vs Edge Operational Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Set parameters
    n = 2
    m = 2
    num_layers = 6
    x_crit = 0.843750000000043
    num_samples = 100
    end_time = 10.0
    sampling_interval = 0.1
    
    # Create x values
    x_values = create_x_values()
    
    # Run sweep
    sweep_results = run_correlation_sweep(
        n=n, m=m, num_layers=num_layers, x_values=x_values,
        num_samples=num_samples, end_time=end_time,
        sampling_interval=sampling_interval
    )
    
    # Plot results
    plot_correlation_sweep(sweep_results, x_crit=x_crit,
        save_path='correlation_sweep.png')
    
    # Save data to CSV
    import pandas as pd
    df = pd.DataFrame({
        'x': x_values,
        'random_avg': sweep_results['random_averages'],
        'random_std': sweep_results['random_stds'],
        'stratified_avg': sweep_results['stratified_averages'],
        'stratified_std': sweep_results['stratified_stds']
    })
    df.to_csv('correlation_sweep_results.csv', index=False)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()