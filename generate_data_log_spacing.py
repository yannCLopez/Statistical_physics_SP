import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.stats import linregress
from scipy.optimize import minimize_scalar

# Set precision for mpmath
mp.dps = 50  # Set the decimal precision to 50 digits

def compute_rho_precise(n, m, x, tol=mp.mpf('1e-12'), max_iter=1000000):
    """
    Compute rho(x) using high precision, iterating F(k) until convergence.
    """
    F_prev = mp.mpf(1.0)  # F(0) = 1 with high precision
    for _ in range(max_iter):
        F_next = (1 - (1 - x * F_prev)**n)**m
        if abs(F_next - F_prev) < tol:
            return F_next
        F_prev = F_next
    return F_prev

def compute_P_K_precise_log_spaced(n, m, x, k_max, base=1.1, tol=mp.mpf('1e-12')):
    """
    Compute the probability distribution P(K = k) for logarithmically spaced k values.

    Args:
        n (int): Number of children per input type.
        m (int): Number of input types.
        x (mp.mpf): High precision probability that an edge is operational.
        k_max (int): Maximum value of k to compute P(K = k).
        base (float): Base for logarithmic spacing.
        tol (mp.mpf): Tolerance for convergence.

    Returns:
        dict: A dictionary where keys are integers k and values are P(K = k).
    """
    # Step 1: Compute rho(x) with high precision
    rho = compute_rho_precise(n, m, x, tol=tol)

    # Step 2: Generate logarithmically spaced k values
    k_values = [0] + [int(base**i) for i in range(int(np.log(k_max) / np.log(base)) + 1)]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort

    # Step 3: Compute F(k) for the selected k values
    F = {0: mp.mpf(1.0)}  # F(0) = 1
    for k in k_values[1:]:
        Fk = (1 - (1 - x * F[k_values[k_values.index(k)-1]])**n)**m #k_values[k_values.index(k)-1] gets the previous k value in our logarithmically spaced list.
        F[k] = Fk
        # Early stopping if Fk converges to rho(x)
        if abs(Fk - rho) < tol:
            for remaining_k in k_values[k_values.index(k)+1:]:
                F[remaining_k] = rho
            break

    # Step 4: Compute P(K = k) = F(k) - F(k_next) for the selected k values
    P_K = {}
    for i, k in enumerate(k_values[:-1]):
        k_next = k_values[i+1]
        P_K[k] = F[k] - F[k_next]
    P_K[k_values[-1]] = F[k_values[-1]] - rho

    return P_K

def chi(r, m, n):
    r, m, n = map(mp.mpf, (r, m, n))
    numerator = 1 - (1 - r**(1/m))**(1/n)
    return float(numerator / r) if r != 0 else float('inf')

def compute_critical_values(m, n, precision=50):
    mp.dps = precision  # Set decimal precision
    
    def chi_wrapper(r):
        return chi(r, m, n)
    
    # Use SciPy's minimize_scalar to find the global minimum
    result = minimize_scalar(chi_wrapper, bounds=(1e-10, 1), method='bounded')
    
    r_crit = mp.mpf(result.x)
    x_crit = mp.mpf(chi_wrapper(r_crit))
    
    return r_crit, x_crit

# Example usage
n = 2              # Number of children per input type
m = 3              # Number of input types
r_crit, x_crit = compute_critical_values(m, n)

print(f"For m = {m} and n = {n}:")
print(f"r_crit = {mp.nstr(r_crit, 12)}")
print(f"x_crit = {mp.nstr(x_crit, 12)}")

# Parameters for high-precision computation
x = x_crit  # High precision value for x
k_max = 10**6       # Compute P(K=k) up to k=10^6
base = 1.01          # Base for logarithmic spacing

# Compute P(K = k) with high precision and logarithmic spacing
P_K_precise_log_spaced = compute_P_K_precise_log_spaced(n, m, x, k_max, base)

# Normalize the distribution to condition on finite K (ignore P(K = infinity))
total_finite_sum = sum(P_K_precise_log_spaced.values())

# Normalize P(K = k) by the total finite sum
P_K_conditional_log_spaced = {k: P_K_precise_log_spaced[k] / total_finite_sum for k in P_K_precise_log_spaced}

def mpf_to_str(value):
    """Convert mpf object to string representation."""
    return str(value)

# Save the output in a file with the value of x in the file name
folder_path = "/Users/yanncalvolopez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Career/RA Ben/Statistical physics of supply chains"
output_file = f"{folder_path}/P_K_log_spaced_m{m}_n{n}_x{mpf_to_str(x)}.txt"

# Save the output
with open(output_file, 'w') as file:
    for k, prob in P_K_conditional_log_spaced.items():
        file.write(f"{k}\t{mpf_to_str(prob)}\n")

# Prepare data for log-log plotting
k_values_precise = np.array(list(P_K_conditional_log_spaced.keys()))
P_K_values_precise = np.array([P_K_conditional_log_spaced[k] for k in k_values_precise])

# Convert k values and P(K) values from mpmath.mpf to floats for compatibility with numpy
k_values_filtered = np.array([float(k) for k in k_values_precise if k > 10**3])
P_K_values_filtered = np.array([float(P_K_conditional_log_spaced[k]) for k in k_values_precise if k > 10**3])

# Apply log transformation
log_k = np.log(k_values_filtered)
log_P_K = np.log(P_K_values_filtered)

# Perform linear regression on the log-log data
slope, intercept, r_value, p_value, std_err = linregress(log_k, log_P_K)

# Extract the power law parameters
b = -slope  # Slope of the log-log plot gives -b
a = np.exp(intercept)  # Intercept gives log(a), so a = exp(intercept)

# Display the results
print(f"Estimated power-law parameters:")
print(f"a = {a:.4e}")
print(f"b = {b:.4f}")

# Plot the filtered data and fitted line
plt.figure(figsize=(10, 6))
plt.loglog(k_values_filtered, P_K_values_filtered, marker='o', linestyle='', color='b', label="P(K = k)")
plt.loglog(k_values_filtered, a / (k_values_filtered**b), linestyle='--', color='r', label=f"Fitted Power Law: a/k^b")
plt.xlabel("k")
plt.ylabel("P(K = k)")
plt.title("Log-Log Plot of P(K = k) with Power Law Fit (k > 10^3)")
plt.legend()
plt.grid(True)
plt.show()
