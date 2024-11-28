import numpy as np
from itertools import combinations
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from poisson_simulation_vectorized_BEN import SupplyNetwork, PoissonSimulation
from typing import List, Dict, Tuple, Set, Optional, Literal

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
        
        # Breadth-First Search to assign tiers
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
    
    def sample_nodes_fully_random(self) -> Set[int]:
        """Randomly sample nodes from all tiers without restriction"""
        all_nodes = list(self.get_all_nodes())
        num_samples = min(self.num_samples, len(all_nodes))
        return set(random.sample(all_nodes, num_samples))
    
    def sample_nodes_tier_restricted(self, selected_tiers: List[int]) -> Set[int]:
        """
        Randomly sample nodes only from specified tiers, without stratification
        
        Args:
            selected_tiers: List of tier numbers to sample from
        """
        tiers = self.get_nodes_by_tier()
        available_nodes = []
        
        # Collect all nodes from selected tiers
        for tier in selected_tiers:
            if tier in tiers:
                available_nodes.extend(list(tiers[tier]))
            else:
                print(f"Warning: Tier {tier} does not exist in the network")
        
        if not available_nodes:
            raise ValueError(f"None of the selected tiers {selected_tiers} exist in the network")
        
        # Sample randomly from the pool of available nodes
        num_samples = min(self.num_samples, len(available_nodes))
        return set(random.sample(available_nodes, num_samples))
    
    def sample_nodes_stratified(self, selected_tiers: Optional[List[int]] = None) -> Set[int]:
        """
        Sample nodes with stratification by tier
        
        Args:
            selected_tiers: List of tier numbers to sample from. If None, sample from all tiers.
        """
        tiers = self.get_nodes_by_tier()
        
        # Filter tiers if selected_tiers is provided
        if selected_tiers is not None:
            tiers = {tier: nodes for tier, nodes in tiers.items() if tier in selected_tiers}
            if not tiers:
                raise ValueError(f"None of the selected tiers {selected_tiers} exist in the network")
        
        # Calculate total available nodes
        total_available = sum(len(nodes) for nodes in tiers.values())
        samples_per_tier = max(1, self.num_samples // len(tiers))
        stratified_samples = set()
        
        # First pass: try to get equal samples from each tier
        for tier_nodes in tiers.values():
            num_tier_samples = min(samples_per_tier, len(tier_nodes))
            stratified_samples.update(random.sample(list(tier_nodes), num_tier_samples))
        
        # If we need more samples and have available nodes in our selected tiers
        remaining_samples = self.num_samples - len(stratified_samples)
        if remaining_samples > 0:
            # Only consider nodes from selected tiers
            available_nodes = set()
            for tier in tiers:
                available_nodes.update(tiers[tier])
            remaining_nodes = list(available_nodes - stratified_samples)
            
            if remaining_nodes:
                additional_samples = random.sample(remaining_nodes,
                                                min(remaining_samples, len(remaining_nodes)))
                stratified_samples.update(additional_samples)
        
        # Add warning if we couldn't get the requested number of samples
        if len(stratified_samples) < self.num_samples:
            print(f"\nWarning: Could only sample {len(stratified_samples)} nodes "
                f"(requested {self.num_samples}) from the selected tiers "
                f"due to limited number of available nodes")
        
        return stratified_samples


    def record_node_states(self, time_point: float):
        """Record the states of sampled nodes at a given time point"""
        states = {node: self.network.node_states[node] for node in self.sample_nodes}
        self.node_states_history.append(states)
        self.time_points.append(time_point)
    
    def calculate_correlations(self) -> Tuple[float, List[float]]:
        """Calculate average correlation between pairs of sampled nodes"""
        node_series = {}
        for node in self.sample_nodes:
            series = [1 if states[node] else 0 for states in self.node_states_history]
            node_series[node] = series
            
        correlations = []
        skipped_pairs = 0
        
        for node1, node2 in combinations(self.sample_nodes, 2):
            series1 = node_series[node1]
            series2 = node_series[node2]
            
            # Check if either series is constant
            if len(set(series1)) == 1 and len(set(series2)) == 1:
                skipped_pairs += 1
                continue
            elif len(set(series1)) == 1 or len(set(series2)) == 1:
                skipped_pairs += 1
                continue
                
            correlation = np.corrcoef(series1, series2)[0, 1]
            if np.isnan(correlation):
                skipped_pairs += 1
                continue
            correlations.append(correlation)
        
        if not correlations:
            raise ValueError(f"No correlations were calculated - all {skipped_pairs} pairs were skipped or resulted in NaN")
            
        print(f"Successfully calculated correlations for {len(correlations)} pairs, skipped {skipped_pairs} pairs")
        avg_correlation = np.mean(correlations)
        return avg_correlation, correlations

    def run_analysis(self, end_time: float, sampling_interval: float, 
                    sampling_method: Literal['fully_random', 'tier_restricted', 'stratified'] = 'fully_random',
                    selected_tiers: Optional[List[int]] = None) -> Tuple[float, List[float]]:
        """
        Run full correlation analysis
        
        Args:
            end_time: Time to run simulation until
            sampling_interval: Time between samples
            sampling_method: Method to use for sampling nodes:
                           'fully_random' - sample from all tiers randomly
                           'tier_restricted' - sample only from selected tiers randomly
                           'stratified' - stratified sampling from selected tiers
            selected_tiers: List of tier numbers to sample from (required for tier_restricted and optional for stratified)
        """
        # Sample nodes based on method
        if sampling_method == 'fully_random':
            self.sample_nodes = self.sample_nodes_fully_random()
        elif sampling_method == 'tier_restricted':
            if selected_tiers is None:
                raise ValueError("selected_tiers must be provided when using tier_restricted sampling")
            self.sample_nodes = self.sample_nodes_tier_restricted(selected_tiers)
        elif sampling_method == 'stratified':
            self.sample_nodes = self.sample_nodes_stratified(selected_tiers)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Clear previous history
        self.node_states_history = []
        self.time_points = []
        
        # Run simulation and record states
        sim = PoissonSimulation(self.network)
        current_time = 0
        
        while current_time < end_time:
            self.record_node_states(current_time)
            current_time += sampling_interval
            sim.run_until(current_time)
        
        return self.calculate_correlations()

def run_correlation_study(n: int, m: int, num_layers: int, x: float, 
                         num_samples: int, end_time: float,
                         sampling_interval: float,
                         selected_tiers: Optional[List[int]] = None) -> dict:
    """Run correlation analysis with all three sampling methods"""
    # Create network
    network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
    analyzer = NodeCorrelationAnalyzer(network, num_samples)
    
    results = {}
    
    # Run fully random sampling (all tiers)
    fully_random_avg, fully_random_corrs = analyzer.run_analysis(
        end_time, sampling_interval, sampling_method='fully_random')
    results['fully_random'] = {
        'average': fully_random_avg,
        'correlations': fully_random_corrs
    }
    
    if selected_tiers is not None:
        # Run tier-restricted random sampling
        tier_random_avg, tier_random_corrs = analyzer.run_analysis(
            end_time, sampling_interval, 
            sampling_method='tier_restricted',
            selected_tiers=selected_tiers)
        results['tier_restricted'] = {
            'average': tier_random_avg,
            'correlations': tier_random_corrs
        }
        
        # Run stratified sampling
        strat_avg, strat_corrs = analyzer.run_analysis(
            end_time, sampling_interval,
            sampling_method='stratified',
            selected_tiers=selected_tiers)
        results['stratified'] = {
            'average': strat_avg,
            'correlations': strat_corrs
        }
    
    return results

def plot_correlation_sweep(results: dict, x_crit: float = None, save_path: str = None,
                         selected_tiers: Optional[List[int]] = None):
    """Plot how correlations vary with x for both sampling methods"""
    plt.figure(figsize=(12, 8))
    
    x_values = results['x_values']
    
    # Plot average correlations with standard deviation bands
    if 'fully_random_averages' in results:
        plt.plot(x_values, results['fully_random_averages'], 'b-', 
                label='Fully Random (All Tiers)', linewidth=2)
        plt.fill_between(x_values, 
                        results['fully_random_averages'] - results['fully_random_stds'],
                        results['fully_random_averages'] + results['fully_random_stds'],
                        color='blue', alpha=0.1)
    
    if 'tier_restricted_averages' in results:
        plt.plot(x_values, results['tier_restricted_averages'], 'g-',
                label=f'Tier-Restricted Random (Tiers {selected_tiers})', linewidth=2)
        plt.fill_between(x_values,
                        results['tier_restricted_averages'] - results['tier_restricted_stds'],
                        results['tier_restricted_averages'] + results['tier_restricted_stds'],
                        color='green', alpha=0.1)
    
    if 'stratified_averages' in results:
        plt.plot(x_values, results['stratified_averages'], 'r-',
                label=f'Stratified (Tiers {selected_tiers})', linewidth=2)
        plt.fill_between(x_values,
                        results['stratified_averages'] - results['stratified_stds'],
                        results['stratified_averages'] + results['stratified_stds'],
                        color='red', alpha=0.1)
    
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
    
    # Add note about limited samples if applicable
    if hasattr(results, 'limited_samples') and results['limited_samples']:
        plt.figtext(0.02, 0.98, 'Note: Some points limited by available nodes in selected tiers',
                   fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_distributions(results: dict, selected_tiers: Optional[List[int]] = None, 
                                 save_path: str = None):
    """Plot distributions of correlations for all sampling methods"""
    num_methods = len(results)
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 6))
    if num_methods == 1:
        axes = [axes]
    
    colors = {'fully_random': 'blue', 'tier_restricted': 'green', 'stratified': 'orange'}
    titles = {
        'fully_random': 'Fully Random\n(All Tiers)',
        'tier_restricted': f'Tier-Restricted Random\n(Tiers {selected_tiers})',
        'stratified': f'Stratified\n(Tiers {selected_tiers})'
    }
    
    for ax, (method, data) in zip(axes, results.items()):
        correlations = data['correlations']
        actual_samples = data.get('actual_samples', len(correlations))
        requested_samples = data.get('requested_samples', actual_samples)
        
        n_bins = 30
        ax.hist(correlations, bins=n_bins, alpha=0.7, 
               label=method, density=True, color=colors[method])
        
        mean = np.mean(correlations)
        std = np.std(correlations)
        ax.axvline(mean, color=colors[method], linestyle='--', 
                  label=f'Mean: {mean:.3f}\nStd: {std:.3f}')
        ax.axvline(mean + std, color=colors[method], linestyle=':', alpha=0.5)
        ax.axvline(mean - std, color=colors[method], linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Pairwise Correlation Coefficient')
        ax.set_ylabel('Density')
        ax.set_title(titles[method])
        ax.legend()
        
        # Add sample size information
        info_text = f'Number of pairs: {len(correlations):,}'
        if actual_samples < requested_samples and method != 'fully_random':
            info_text += f'\n(Limited by available nodes:\n{actual_samples}/{requested_samples} samples)'
        
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle('Distribution of Pairwise Node State Correlations', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

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
                         num_samples: int, end_time: float,
                         sampling_interval: float,
                         selected_tiers: Optional[List[int]] = None) -> dict:
    """Run correlation analysis for multiple x values with all sampling methods"""
    # Initialize arrays for each sampling method
    fully_random_avgs = []
    fully_random_stds = []
    tier_restricted_avgs = []
    tier_restricted_stds = []
    stratified_avgs = []
    stratified_stds = []
    limited_samples = False
    
    total_runs = len(x_values)
    
    for i, x in enumerate(x_values):
        print(f"\nAnalyzing x = {x:.3f} ({i+1}/{total_runs})")
        
        # Create network
        network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
        analyzer = NodeCorrelationAnalyzer(network, num_samples)
        
        # Run fully random sampling (all tiers)
        fully_random_avg, fully_random_corrs = analyzer.run_analysis(
            end_time, sampling_interval, sampling_method='fully_random')
        fully_random_avgs.append(fully_random_avg)
        fully_random_stds.append(np.std(fully_random_corrs))
        
        if selected_tiers is not None:
            # Run tier-restricted sampling
            try:
                tier_random_avg, tier_random_corrs = analyzer.run_analysis(
                    end_time, sampling_interval, 
                    sampling_method='tier_restricted',
                    selected_tiers=selected_tiers)
                tier_restricted_avgs.append(tier_random_avg)
                tier_restricted_stds.append(np.std(tier_random_corrs))
            
                # Run stratified sampling
                strat_avg, strat_corrs = analyzer.run_analysis(
                    end_time, sampling_interval,
                    sampling_method='stratified',
                    selected_tiers=selected_tiers)
                stratified_avgs.append(strat_avg)
                stratified_stds.append(np.std(strat_corrs))
                
                # Check if we got fewer samples than requested
                if (len(tier_random_corrs) < (num_samples * (num_samples - 1) // 2) or
                    len(strat_corrs) < (num_samples * (num_samples - 1) // 2)):
                    limited_samples = True
                
            except ValueError as e:
                print(f"Warning: {e}")
                # If sampling fails, use NaN for this point
                tier_restricted_avgs.append(np.nan)
                tier_restricted_stds.append(np.nan)
                stratified_avgs.append(np.nan)
                stratified_stds.append(np.nan)
        
        # Print intermediate results
        print(f"Fully random sampling average correlation: {fully_random_avg:.3f}")
        if selected_tiers is not None:
            print(f"Tier-restricted sampling average correlation: {tier_random_avg:.3f}")
            print(f"Stratified sampling average correlation: {strat_avg:.3f}")
    
    results = {
        'x_values': x_values,
        'fully_random_averages': np.array(fully_random_avgs),
        'fully_random_stds': np.array(fully_random_stds),
    }
    
    if selected_tiers is not None:
        results.update({
            'tier_restricted_averages': np.array(tier_restricted_avgs),
            'tier_restricted_stds': np.array(tier_restricted_stds),
            'stratified_averages': np.array(stratified_avgs),
            'stratified_stds': np.array(stratified_stds),
            'limited_samples': limited_samples
        })
    
    return results

def main():
    # Set parameters
    n = 2
    m = 2
    num_layers = 6
    x_crit = 0.843750000000043
    num_samples = 100
    end_time = 30.0
    sampling_interval = 0.1
    
    # Specify tiers to sample from
    selected_tiers = [1]  # Only sample from tiers 0, 1, and 2
    
    # Create x values
    x_values = create_x_values()
    
    # Run sweep
    sweep_results = run_correlation_sweep(
        n=n, m=m, num_layers=num_layers, x_values=x_values,
        num_samples=num_samples, end_time=end_time,
        sampling_interval=sampling_interval,
        selected_tiers=selected_tiers
    )
    
    # Plot results
    plot_correlation_sweep(sweep_results, x_crit=x_crit,
        save_path='correlation_sweep.png',
        selected_tiers=selected_tiers)
    
    # Save data to CSV
    import pandas as pd
    df = pd.DataFrame({
        'x': x_values,
        'fully_random_avg': sweep_results['fully_random_averages'],
        'fully_random_std': sweep_results['fully_random_stds'],
        'tier_restricted_avg': sweep_results['tier_restricted_averages'],
        'tier_restricted_std': sweep_results['tier_restricted_stds'],
        'stratified_avg': sweep_results['stratified_averages'],
        'stratified_std': sweep_results['stratified_stds']
    })
    df.to_csv('correlation_sweep_results.csv', index=False)

if __name__ == "__main__":
    main()
