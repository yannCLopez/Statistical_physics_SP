# Reimport necessary libraries and re-run the plot since the environment was reset
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.stats import linregress
from scipy.optimize import minimize_scalar




# Redefine high-precision functions

# Set precision for mpmath
mp.dps = 50  # Set the decimal precision to 50 digits

def compute_rho_precise(n, m, x, tol=mp.mpf('1e-12'), max_iter=1000000):
    """
    Compute rho(x) using high precision, iterating F(k) until convergence.
    
    Args:
        n (int): Number of children per input type.
        m (int): Number of input types.
        x (mp.mpf): High precision probability that an edge is operational.
        tol (mp.mpf): Tolerance for convergence.
        max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        mp.mpf: The reliability rho(x) of the infinite tree.
    """
    F_prev = mp.mpf(1.0)  # F(0) = 1 with high precision
    for _ in range(max_iter):
        F_next = (1 - (1 - x * F_prev)**n)**m
        if abs(F_next - F_prev) < tol:
            return F_next
        F_prev = F_next
    return F_prev

def compute_P_K_precise(n, m, x, k_max, tol=mp.mpf('1e-12')):
    """
    Compute the probability distribution P(K = k) for large k with high precision.

    Args:
        n (int): Number of children per input type.
        m (int): Number of input types.
        x (mp.mpf): High precision probability that an edge is operational.
        k_max (int): Maximum value of k to compute P(K = k).
        tol (mp.mpf): Tolerance for convergence.

    Returns:
        dict: A dictionary where keys are integers k (0 <= k <= k_max)
              and values are P(K = k).
    """
    # Step 1: Compute rho(x) with high precision
    rho = compute_rho_precise(n, m, x, tol=tol)

    # Step 2: Compute F(k) for k = 0 to k_max
    F = [mp.mpf(1.0)]  # F(0) = 1
    for k in range(1, k_max + 2):
        Fk = (1 - (1 - x * F[k-1])**n)**m
        F.append(Fk)
        # Early stopping if Fk converges to rho(x)
        if abs(Fk - rho) < tol:
            for remaining_k in range(k + 1, k_max + 2):
                F.append(rho)
            break

    # Step 3: Compute P(K = k) = F(k) - F(k + 1) for k = 0 to k_max
    P_K = {}
    for k in range(0, k_max + 1):
        P_K[k] = F[k] - F[k + 1] if k + 1 < len(F) else F[k] - rho

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
m = 2               # Number of input types
r_crit, x_crit = compute_critical_values(m, n)

print(f"For m = {m} and n = {n}:")
print(f"r_crit = {mp.nstr(r_crit, 12)}")
print(f"x_crit = {mp.nstr(x_crit, 12)}")

# Parameters for high-precision computation

#x = mp.mpf('0.86')  # High precision value for x
#x = mp.mpf('0.84375')  # High precision value for x
x = x_crit  # High precision value for x
k_max = 10**6       # Compute P(K=k) up to k=10^6

# Compute P(K = k) with high precision
P_K_precise_large_k = compute_P_K_precise(n, m, x, k_max)

# Normalize the distribution to condition on finite K (ignore P(K = infinity))
total_finite_sum_large_k = sum(P_K_precise_large_k.values())

# Normalize P(K = k) by the total finite sum
P_K_conditional_large_k = {k: P_K_precise_large_k[k] / total_finite_sum_large_k for k in P_K_precise_large_k}

# Save the output in a file with the value of x in the file name
output_file = f"/Users/benjamingolub/Downloads/stat_phys_snff/P_K_{str(x)}.txt"
output_file = f"/Users/yanncalvolopez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Career/RA Ben/Statistical physics of supply chains/P_K_{str(x)}.txt"
with open(output_file, 'w') as file:
    for k, p in P_K_conditional_large_k.items():
        file.write(f"{k}\t{p}\n")
print(f"Output saved in file: {output_file}")


# Prepare data for log-log plotting
k_values_precise = np.array(list(P_K_conditional_large_k.keys()))
P_K_values_precise = np.array([P_K_conditional_large_k[k] for k in k_values_precise])

# Convert k values and P(K) values from mpmath.mpf to floats for compatibility with numpy
k_values_filtered = np.array([float(k) for k in k_values_precise if k > 10**5])
P_K_values_filtered = np.array([float(P_K_conditional_large_k[k]) for k in k_values_precise if k > 10**5])

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
plt.loglog(k_values_filtered, P_K_values_filtered, marker='o', linestyle='-', color='b', label="P(K = k)")
plt.loglog(k_values_filtered, a / (k_values_filtered**b), linestyle='--', color='r', label=f"Fitted Power Law: a/k^b")
plt.xlabel("k")
plt.ylabel("P(K = k)")
plt.title("Log-Log Plot of P(K = k) with Power Law Fit (k > 10^3)")
plt.legend()
plt.grid(True)
plt.show()
