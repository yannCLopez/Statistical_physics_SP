import numpy as np
from collections import defaultdict
import heapq
from dataclasses import dataclass
import random
import string
from typing import List, Tuple, Dict
from playsound import playsound


@dataclass(order=True)
class Event:
    time: float
    edge: tuple
    priority: int = 0  # For tiebreaking in heapq

class SupplyNetwork:
    def __init__(self, n, m, num_layers, x):
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
        
        self.build_tree()
        
        # Initialize ALL edges as operational (state = 1) at t=0
        for edge in self.edges:
            self.edge_states[edge] = 1

    def create_path_label(self, parent_id: int, type_idx: int, child_idx: int) -> str:
        #checked
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
        #check
        if parent_id == 0:  # Parent is root
            return f"{self.type_labels[type_idx]}.{child_idx + 1}"
        parent_label = self.node_labels[parent_id]
        return f"{parent_label}.{self.type_labels[type_idx]}.{child_idx + 1}"

    def build_tree(self):
        #checked
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
                        print("dictionary: ", self.node_labels)
                        self.label_to_id[path_label] = current_id
                        
                        # Add edge and child info
                        self.edges.add((current_id, parent))
                        self.children[parent][type_idx].append(current_id)
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

    def check_functionality(self) -> bool:
        #partially checked (with AI)
        #revoir logique
        """
        Note: iterations are not in time. 
        Check functionality of the network using bottom-up approach.
        node v is in F_k if, for each input type, it has at least one operational edge 
        to a child node that was functional in the previous iteration (F_{k-1}).
        
        Returns:
        --------
        bool
            True if root is functional, False otherwise
        """
        # Get all nodes
        V = set()
        for parent in self.children:
            for type_children in self.children[parent].values():
                V.update(type_children)
        V.add(0)  # Add root
        
        # F_0 = V (initially assume all nodes are functional)
        F_prev = V.copy()
        
        # Iterate up to depth+1 times (as per original algorithm)
        for k in range(1, self.num_layers + 2):
            # Create new F_k set for this iteration
            F_k = set()
            
            # Check each node
            for v in V:
                # Leaf nodes are always functional
                if not self.children[v]:
                    F_k.add(v)
                    continue
                    
                # Check if node has operational edges to functional children
                has_all_types = True
                for type_idx in range(self.m):
                    has_functional_input = False
                    for child in self.children[v][type_idx]:
                        if child in F_prev and self.edge_states[(child, v)] == 1:
                            has_functional_input = True
                            break
                    if not has_functional_input:
                        has_all_types = False
                        break
                
                if has_all_types:
                    F_k.add(v)
            
            # If no changes occurred, stop iterating
            if F_k == F_prev:
                break
                
            # Update F_{k-1} for next iteration
            F_prev = F_k
        
        return 0 in F_k

    def debug_check_functionality(self) -> bool:
        """Debug version of check_functionality with detailed output"""
        V = set()
        for parent in self.children:
            for type_children in self.children[parent].values():
                V.update(type_children)
        V.add(0)
        
        # Start with leaf nodes
        functional_nodes = set()
        for v in V:
            if not self.children[v]:
                functional_nodes.add(v)
        #        print(f"Leaf node {self.get_node_label(v)} marked as functional")
        
        iteration = 0
        changed = True
        while changed:
            changed = False
            iteration += 1
            print(f"\nIteration {iteration}:")
            
            for v in V:
                if v in functional_nodes:
                    continue
                
          #      print(f"\nChecking node {self.get_node_label(v)}:")
                has_all_types = True
                
                for type_idx in range(self.m):
            #        print(f"  Checking type {self.type_labels[type_idx]} inputs:")
                    has_functional_input = False
                    
                    for child in self.children[v][type_idx]:
                        is_functional = child in functional_nodes
                        edge_state = self.edge_states[(child, v)]
            #           print(f"    Child {self.get_node_label(child)}: "
            #                 f"Functional: {is_functional}, Edge: {edge_state}")
                        
                        if is_functional and edge_state == 1:
                            has_functional_input = True
            #                print(f"    Found functional input from {self.get_node_label(child)}")
                            break
                            
                    if not has_functional_input:
            #            print(f"    No functional input found for type {self.type_labels[type_idx]}")
                        has_all_types = False
                        break
                
                if has_all_types and v not in functional_nodes:
                    functional_nodes.add(v)
                    changed = True
                    #print(f"  Node {self.get_node_label(v)} is now functional")
                #elif not has_all_types:
                    #print(f"  Node {self.get_node_label(v)} is not functional")
            
            #if changed:
                #print(f"\nFunctional nodes after iteration {iteration}:")
                #for v in functional_nodes:
                #   print(f"  {self.get_node_label(v)}")
        
        return 0 in functional_nodes

class PoissonSimulation:
    def __init__(self, network):
        self.network = network
        self.current_time = 0
        self.event_queue = []
        initial_state = self.network.check_functionality()
        self.root_state_changes = [(0.0, initial_state)]  # Record initial state
        self.last_root_state = initial_state
        
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
        while self.event_queue and self.current_time < end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if self.current_time >= end_time:
                break
                
            # Toggle edge state
            edge = event.edge
            self.network.edge_states[edge] = 1 - self.network.edge_states[edge]
            
            # Check if root state changed
            new_root_state = self.network.check_functionality()
            if new_root_state != self.last_root_state:
                self.root_state_changes.append((self.current_time, new_root_state))
                self.last_root_state = new_root_state
            
            # Schedule next event for this edge
            self.schedule_next_event(edge)
            
        return self.root_state_changes
def analyze_root_state_changes(root_state_changes: List[Tuple[float, bool]]) -> dict:
    """
    Analyze the root state changes to provide useful statistics
    """
    if len(root_state_changes) <= 1:  # Changed from not root_state_changes
        return {
            "number_of_changes": 0,
            "time_operational": root_state_changes[0][1] if root_state_changes else 0,
            "time_non_operational": not root_state_changes[0][1] if root_state_changes else 0,
            "average_operational_period": 0,
            "average_non_operational_period": 0
        }
    
    stats = {
        "number_of_changes": len(root_state_changes) - 1,
        "state_changes": root_state_changes
    }
    
    # Calculate periods for each state
    operational_periods = []
    non_operational_periods = []
    
    for i in range(len(root_state_changes) - 1):
        duration = root_state_changes[i + 1][0] - root_state_changes[i][0]
        if root_state_changes[i][1]:  # If state was operational
            operational_periods.append(duration)
        else:
            non_operational_periods.append(duration)
    
    stats["average_operational_period"] = np.mean(operational_periods) if operational_periods else 0
    stats["average_non_operational_period"] = np.mean(non_operational_periods) if non_operational_periods else 0
    
    return stats

def run_simulation(n, m, num_layers, x, end_time):
    network = SupplyNetwork(n=n, m=m, num_layers=num_layers, x=x)
    
    # Verify initial conditions
    print("\nInitial conditions verification:")
    print("All edges operational:", all(state == 1 for state in network.edge_states.values()))
    
    print("\nChecking initial functionality propagation:")
    initial_functionality = network.check_functionality()
    print(f"\nRoot functional at t=0: {initial_functionality}")
    
    if not initial_functionality:
        raise AssertionError("Root must be functional at t=0 when all edges are operational")
    
    sim = PoissonSimulation(network)
    root_state_changes = sim.run_until(end_time)
    stats = analyze_root_state_changes(root_state_changes)
    
    return sim, root_state_changes, stats

if __name__ == "__main__":
    sim, state_changes, stats = run_simulation(n=2, m=2, num_layers=3, x=0.843750000000043, end_time=100)
    
    print("\nRoot state changes:")
    print(f"Initial state at t=0: {'Operational' if state_changes[0][1] else 'Non-operational'}")
    for time, state in state_changes[1:]:  # Skip initial state in the loop
        print(f"t={time:.3f}: {'Operational' if state else 'Non-operational'}")
    
    print("\nStatistics:")
    print(f"Number of state changes: {stats['number_of_changes']}")
    print(f"Average operational period: {stats['average_operational_period']:.3f}")
    print(f"Average non-operational period: {stats['average_non_operational_period']:.3f}")

   #playsound('/Users/yanncalvolopez/Documents/Ringtones/Impro tiersen.2.mp3')