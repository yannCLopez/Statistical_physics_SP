import numpy as np
from collections import defaultdict
import heapq
from dataclasses import dataclass
import random
import string
from typing import List, Tuple, Dict
import os
import sys  # Added for progress bar updates

@dataclass(order=True)
class Event:
    time: float
    edge: tuple
    priority: int = 0  # For tiebreaking in heapq

class SupplyNetwork:
    def _init_(self, n, m, num_layers, x):
        """
        Initialize supply network with n children per type, m types, and specified number of layers.
        x is the desired steady-state probability of an edge being operational.
        At t=0, all edges are operational.
        
        Parameters:
        -----------
        n : int
            Number of children per type
        m : int
            Number of types
        num_layers : int
            Number of layers in the network
        x : float
            Steady-state probability of an edge being operational
        """
        self.n = n  # number of children per type
        self.m = m  # number of types
        self.num_layers = num_layers
        self.lambda_off = 1.0  # normalized
        self.lambda_on = (x * self.lambda_off) / (1 - x)  # derived from steady state equation
        
        # Create type labels (a, b, c, ...)
        self.type_labels = list(string.ascii_lowercase[:m])
        
        # Build the tree structure
        self.edges = set()
        self.edge_states = {}  # 1 for operational, 0 for non-operational
        self.children = defaultdict(lambda: defaultdict(list))
        self.node_labels = {}  # Maps node ID to path label
        self.label_to_id = {}  # Maps path label to node ID
        self.parents = {}      # Maps node ID to its parent node ID
        self.node_states = {}  # Maps node ID to its functionality state
        
        self.build_tree()
        
        # Initialize ALL edges as operational (state = 1) at t=0
        for edge in self.edges:
            self.edge_states[edge] = 1 if random.random() < x else 0
        
        # Initialize node states
        self.initialize_node_states()
    
    def create_path_label(self, parent_id: int, type_idx: int, child_idx: int) -> str:
        """
        Create a path label for a node based on its parent's label and its position
        
        Parameters:
        -----------
        parent_id : int
            ID of the parent node
        type_idx : int
            Index of the type (0 to m-1)
        child_idx : int
            Index of the child within its type (0 to n-1)
                
        Returns:
        --------
        str
            Path label for the node
        """
        if parent_id == 0:  # Parent is root
            return f"{self.type_labels[type_idx]}.{child_idx + 1}"
        parent_label = self.node_labels[parent_id]
        return f"{parent_label}.{self.type_labels[type_idx]}.{child_idx + 1}"
    
    def build_tree(self):
        """Build the tree structure with n children per type and m types"""
        nodes = {0}  # Start with root node
        current_id = 1
        
        # Label root node
        self.node_labels[0] = "root"
        self.label_to_id["root"] = 0
        
        # Build layer by layer
        for layer in range(self.num_layers):
            new_nodes = set()
            for parent in nodes:
                for type_idx in range(self.m):
                    for child_idx in range(self.n):
                        # Create node with path label
                        path_label = self.create_path_label(parent, type_idx, child_idx)
                        self.node_labels[current_id] = path_label
                        self.label_to_id[path_label] = current_id
                        
                        # Add edge and child info
                        self.edges.add((current_id, parent))
                        self.children[parent][type_idx].append(current_id)
                        self.parents[current_id] = parent  # Store parent
                        new_nodes.add(current_id)
                        current_id += 1
            nodes = new_nodes

    def get_node_label(self, node_id: int) -> str:
        """Get the path label for a node ID"""
        return self.node_labels.get(node_id, str(node_id))

    def get_node_id(self, label: str) -> int:
        """Get the node ID for a path label"""
        return self.label_to_id.get(label, -1)

    def print_network_structure(self):
        """Print the network structure showing path labels and current states"""
        def print_node(node_id: int, indent: int = 0):
            node_label = self.get_node_label(node_id)
            print("  " * indent + f"Node {node_label}")
            for type_idx in range(self.m):
                children = self.children[node_id][type_idx]
                if children:
                    print("  " * (indent + 1) + f"Type {self.type_labels[type_idx]} children:")
                    for child_id in children:
                        state = self.edge_states[(child_id, node_id)]
                        child_label = self.get_node_label(child_id)
                        print("  " * (indent + 2) + 
                              f"{child_label} -> {node_label} " +
                              f"({'operational' if state else 'non-operational'})")
                        print_node(child_id, indent + 3)
        
        print("Network Structure:")
        print_node(0)
    
    def initialize_node_states(self):
        """Initialize the state of each node in the network"""
        def post_order(node_id):
            # For leaf nodes, they are functional
            if not any(self.children[node_id].values()):
                self.node_states[node_id] = True
                return

            for type_idx in range(self.m):
                for child in self.children[node_id][type_idx]:
                    post_order(child)

            self.node_states[node_id] = self.compute_node_state(node_id)

        post_order(0)  # Start from root

    def compute_node_state(self, node_id):
        """Compute the functionality state of a node based on its children's states"""
        # For leaf nodes, they are functional
        if not any(self.children[node_id].values()):
            return True

        for type_idx in range(self.m):
            has_functional_child = False
            for child in self.children[node_id][type_idx]:
                edge_state = self.edge_states[(child, node_id)]
                child_state = self.node_states[child]
                if edge_state == 1 and child_state:
                    has_functional_child = True
                    break
            if not has_functional_child:
                return False

        return True

    def update_node_states(self, edge):
        """Update the states of nodes along the path from the changed edge to the root"""
        child_id, parent_id = edge

        node_id = parent_id
        while True:
            old_state = self.node_states[node_id]
            new_state = self.compute_node_state(node_id)
            self.node_states[node_id] = new_state
            if new_state == old_state:
                # No change in state, can stop
                break
            if node_id == 0:
                # Reached root
                break
            node_id = self.parents.get(node_id, None)
            if node_id is None:
                break

class PoissonSimulation:
    def _init_(self, network):
        self.network = network
        self.current_time = 0
        self.event_queue = []
        self.last_root_state = self.network.node_states[0]
        self.root_state_changes = [(0.0, self.last_root_state)]  # Record initial state
        
        # Initialize events for all edges
        for edge in self.network.edges:
            self.schedule_next_event(edge)
    
    def schedule_next_event(self, edge):
        """Schedule the next event for an edge based on its current state"""
        current_state = self.network.edge_states[edge]
        rate = self.network.lambda_on if current_state == 0 else self.network.lambda_off
        next_time = self.current_time + np.random.exponential(1/rate)
        heapq.heappush(self.event_queue,
                       Event(time=next_time, edge=edge, priority=random.random()))
    
    def run_until(self, end_time: float) -> List[Tuple[float, bool]]:
        """
        Run simulation until specified end time
        Returns list of root state changes
        """
        last_update_time = 0.0  # Initialize last update time for progress bar
        update_interval = 0.03   # Time interval for progress updates

        while self.event_queue and self.current_time < end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if self.current_time >= end_time:
                break
                
            # Toggle edge state
            edge = event.edge
            self.network.edge_states[edge] = 1 - self.network.edge_states[edge]
            
            # Update node states along the path to root
            self.network.update_node_states(edge)
            
            # Check if root state changed
            new_root_state = self.network.node_states[0]
            if new_root_state != self.last_root_state:
                self.root_state_changes.append((self.current_time, new_root_state))
                self.last_root_state = new_root_state
            
            # Schedule next event for this edge
            self.schedule_next_event(edge)
            
            # Update progress bar if necessary
            while self.current_time >= last_update_time + update_interval:
                last_update_time += update_interval
                progress = min(last_update_time / end_time * 100, 100)
                bar_length = 50  # Length of the progress bar
                filled_length = int(bar_length * progress // 100)
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: |{bar}| {progress:.2f}% ', end='', flush=True)
        
        # Ensure the progress bar reaches 100%
        progress = 100
        bar_length = 50
        bar = '#' * bar_length
        print(f'\rProgress: |{bar}| {progress:.2f}% ')
        
        return self.root_state_changes

def analyze_root_state_changes(root_state_changes: List[Tuple[float, bool]], end_time: float, output_file: str = "down_times.txt") -> dict:
    """
    Analyze the root state changes and write down-time durations to file as they occur
    
    Parameters:
    -----------
    root_state_changes : List[Tuple[float, bool]]
        List of (time, state) pairs recording when root state changed
    end_time : float
        Total simulation time
    output_file : str
        Name of file to write down-time durations to
        
    Returns:
    --------
    dict
        Dictionary containing various statistics
    """
    # Open file in write mode
    with open(output_file, 'w') as f:
        f.write("duration\n")  # Header
        
        if len(root_state_changes) <= 1:
            # If system starts non-operational, write entire time as one down period
            if not root_state_changes[0][1]:
                f.write(f"{end_time}\n")
            
            operational_time = end_time if root_state_changes[0][1]  else 0
            return {
                "number_of_changes": 0,     
                "time_operational": operational_time,          
                "time_non_operational": end_time - operational_time,
                "fraction_time_operational": operational_time / end_time if end_time > 0 else 0,
                "average_operational_period": 0,
                "average_non_operational_period": 0
            }
    
        total_operational_time = 0
        operational_periods = []
        non_operational_periods = []
        
        # Process all periods including the final one
        for i in range(len(root_state_changes)):
            current_time = root_state_changes[i][0]
            current_state = root_state_changes[i][1]
            
            # Get end time for this period
            next_time = end_time if i == len(root_state_changes) - 1 else root_state_changes[i + 1][0]
            period_duration = next_time - current_time
            
            if current_state:
                total_operational_time += period_duration
                if i < len(root_state_changes) - 1:
                    operational_periods.append(period_duration)
            else:
                # Write down-time duration to file immediately
                if i < len(root_state_changes) - 1:
                    f.write(f"{period_duration}\n")
                    non_operational_periods.append(period_duration)
    
    return {
        "number_of_changes": len(root_state_changes) - 1,
        "time_operational": total_operational_time,
        "time_non_operational": end_time - total_operational_time,
        "fraction_time_operational": total_operational_time / end_time if end_time > 0 else 0,
        "average_operational_period": np.mean(operational_periods) if operational_periods else 0,
        "average_non_operational_period": np.mean(non_operational_periods) if non_operational_periods else 0
    }

def calculate_analytical_F(n, m, x, num_layers):
    """
    Calculate F(k) iteratively using the formula:
    F(k+1) = [1 - (1 - xF(k))^n]^m
    
    Parameters:
    -----------
    n : int
        Number of children per type
    m : int
        Number of types
    x : float
        Probability of edge being operational
    num_layers : int
        Number of layers (iterations)
    
    Returns:
    --------
    float
        Final value of F after num_layers iterations
    """
    F = 1.0  # F(0) = 1
    
    for k in range(num_layers):
        F = (1 - (1 - x*F)*n)*m
    
    return F

def run_simulation2(n, m, num_layers, x, end_time, output_file):
    network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
    
    # Verify initial conditions
    print("\nInitial conditions verification:")
    num_operational = sum(state == 1 for state in network.edge_states.values())
    total_edges = len(network.edge_states)
    initial_ratio = num_operational / total_edges
    print(f"Proportion of operational edges: {initial_ratio:.3f} (expected ≈ {x:.3f})")
    
    print("\nChecking initial functionality propagation:")
    initial_functionality = network.node_states[0]
    print(f"Root functional at t=0: {initial_functionality}")
    
    sim = PoissonSimulation(network)
    root_state_changes = sim.run_until(end_time)
    stats = analyze_root_state_changes(root_state_changes, end_time, output_file)
    
    # Calculate analytical result
    analytical_result = calculate_analytical_F(n, m, x, num_layers)
    
    print("\nSimulation Results:")
    print(f"Fraction of time root was operational: {stats['fraction_time_operational']:.3f}")
    print(f"Analytical prediction: {analytical_result:.3f}")
    print(f"Total operational time: {stats['time_operational']:.3f}")
    print(f"Total non-operational time: {stats['time_non_operational']:.3f}")
    print(f"\nDown-time durations have been written to {output_file}")
    
    return sim, root_state_changes, stats

def main():
    output_folder = os.path.dirname(os.path.abspath(_file_))
    n = 2
    m = 2
    num_layers = 11
    #x=0.843750000000043
    #x = 0.79
    x = 0.83
    end_time = 100
    output_file = os.path.join(output_folder, f"down_times_n_{n}m{m}layers{num_layers}x{x}end_time{end_time}.txt")
    
    sim, state_changes, stats = run_simulation2(n, m, num_layers, x, end_time, output_file)
    print(f"Initial state at t=0: {'Operational' if state_changes[0][1] else 'Non-operational'}")
    
    print("\nStatistics:")
    print(f"Number of state changes: {stats['number_of_changes']}")
    print(f"Average operational period: {stats['average_operational_period']:.3f}")
    print(f"Average non-operational period: {stats['average_non_operational_period']:.3f}")

if _name_ == "_main_":
    main()
