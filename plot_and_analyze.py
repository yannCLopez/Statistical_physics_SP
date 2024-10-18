import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.stats import linregress


#x= mp.mpf(.84375)

# Load the output just saved (for a routine where the data has not been generated in the same session)
#input_file = f"/Users/benjamingolub/Downloads/stat_phys_snff/P_K_{str(x)}.txt"
input_file = "/Users/yanncalvolopez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Career/RA Ben/Statistical physics of supply chains/P_K_log_spaced_p2-0.5_3-0.5_q2-0.5_3-0.5_x0.804846771157344.txt"
P_K_conditional_large_k = {}  # Initialize the dictionary
with open(input_file, 'r') as file:
    for line in file:
        k, p = line.strip().split('\t')
        P_K_conditional_large_k[int(k)] = float(p)



# Continue with the rest of the code...

# Prepare data for log-log plotting
k_values_precise = np.array(list(P_K_conditional_large_k.keys()))
P_K_values_precise = np.array([P_K_conditional_large_k[k] for k in k_values_precise])

# Convert k values and P(K) values from mpmath.mpf to floats for compatibility with numpy
k_values_filtered = []
P_K_values_filtered = []

# Filter and convert data: ensure P(K=k) > 0 to avoid log issues
for k in k_values_precise:
    if k > 10**3 and P_K_conditional_large_k[k] > 0:
        k_values_filtered.append(float(k))
        P_K_values_filtered.append(float(P_K_conditional_large_k[k]))

k_values_filtered = np.array(k_values_filtered)
P_K_values_filtered = np.array(P_K_values_filtered)

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