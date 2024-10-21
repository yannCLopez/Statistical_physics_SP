import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
import json
import requests
import base64
import os
import argparse

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
    Compute the probability distribution P(K = k) for logarithmically spaced k values,
    but compute F[k] for all k up to k_max.
    """
    # Step 1: Compute rho(x) with high precision
    rho = compute_rho_precise(n, m, x, tol=tol)

    # Step 2: Generate logarithmically spaced k values
    k_values = [0] + [int(base**i) for i in range(int(np.log(k_max) / np.log(base)) + 1)]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort

    # Step 3: Compute F(k) for all k up to k_max
    F = {0: mp.mpf(1.0)}  # F(0) = 1
    for k in range(1, k_max + 1):
        Fk = (1 - (1 - x * F[k-1])**n)**m
        F[k] = Fk
        # Early stopping if Fk converges to rho(x)
        if abs(Fk - rho) < tol:
            k_max = k
            break

    # Step 4: Compute P(K = k) = F(k) - F(k+1) for the logarithmically spaced k values
    P_K = {}
    for k in k_values:
        if k < k_max:
            P_K[k] = F[k] - F[k+1]
        else:
            P_K[k] = 0
            break

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

def mpf_to_str(value):
    """Convert mpf object to string representation."""
    return str(value)

def save_to_github(content, filename, repo_info):
    owner, repo, path, token = repo_info
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print(f"Attempting to save file: {filename}")
    
    # Check if file already exists
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        existing_file = response.json()
        sha = existing_file["sha"]
        print(f"File {filename} already exists. SHA: {sha}")
    elif response.status_code == 404:
        sha = None
        print(f"File {filename} does not exist. Creating new file.")
    else:
        print(f"Error checking file existence. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

    # Prepare the file content
    content_bytes = content.encode("utf-8")
    base64_bytes = base64.b64encode(content_bytes)
    base64_string = base64_bytes.decode("utf-8")

    data = {
        "message": f"Add/Update {filename}",
        "content": base64_string,
    }
    if sha:
        data["sha"] = sha  # Include SHA to ensure we're updating the existing file

    # Create or update the file
    response = requests.put(url, headers=headers, json=data)
    if response.status_code in [201, 200]:
        print(f"File successfully uploaded to GitHub: {url}")
        return True
    else:
        print(f"Failed to upload file to GitHub. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compute and save P(K) distribution")
    parser.add_argument("--github", help="GitHub repository in the format owner/repo/folder")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument("--local", help="Local directory path")
    args = parser.parse_args()

    if not (args.local or (args.github and args.token)):
        print("Please provide either a GitHub repository (--github and --token) or a local directory (--local)")
        return

    # Determine where to save the files
    if args.github and args.token:
        repo_parts = args.github.split('/')
        if len(repo_parts) < 3:
            print("Please provide a valid GitHub repository in the format owner/repo/folder")
            return
        repo_owner = repo_parts[0]
        repo_name = repo_parts[1]
        folder_path = '/'.join(repo_parts[2:])
        repo_info = (repo_owner, repo_name, folder_path, args.token)
        save_function = lambda content, filename: save_to_github(content, filename, repo_info)
    else:
        folder_path = args.local
        save_function = lambda content, filename: open(os.path.join(folder_path, filename), 'w').write(content)

    # Example usage (keeping the original values)
    n = 3              # Number of children per input type
    m = 3              # Number of input types
    r_crit, x_crit = compute_critical_values(m, n)

    print(f"For m = {m} and n = {n}:")
    print(f"r_crit = {mp.nstr(r_crit, 12)}")
    print(f"x_crit = {mp.nstr(x_crit, 12)}")

    # Parameters for high-precision computation
    x = x_crit  # High precision value for x
    k_max = 10**6  # Compute P(K=k) up to k=10^6
    base = 1.01  # Base for logarithmic spacing

    # Compute P(K = k) with high precision and logarithmic spacing
    P_K_precise_log_spaced = compute_P_K_precise_log_spaced(n, m, x, k_max, base)

    # Normalize the distribution to condition on finite K (ignore P(K = infinity))
    total_finite_sum = sum(P_K_precise_log_spaced.values())

    # Normalize P(K = k) by the total finite sum
    P_K_conditional_log_spaced = {k: P_K_precise_log_spaced[k] / total_finite_sum for k in P_K_precise_log_spaced}

    # Prepare the content to save
    filename = f"P_K_log_spaced_m{m}_n{n}_x{mpf_to_str(x)}.txt"
    content = ""
    for k, prob in P_K_conditional_log_spaced.items():
        content += f"{k}\t{mpf_to_str(prob)}\n"

    # Save the main file
    save_function(content, filename)

    # Save the metadata
   # metadata = {
   #     "m": m,
   #     "n": n,
   #     "x_value": mpf_to_str(x)
   # }
   # metadata_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
   # metadata_content = json.dumps(metadata, indent=2)
   # save_function(metadata_content, metadata_filename)

    #print(f"Files saved: {filename} and {metadata_filename}")

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
    plt.title(f"Log-Log Plot of P(K = k) with Power Law Fit (k > 10^3), m={m}, n={n}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    #plot_filename = filename.rsplit('.', 1)[0] + '_plot.png'
    #plt.savefig(plot_filename)
    #print(f"Plot saved: {plot_filename}")

    #if args.github:
    #    with open(plot_filename, 'rb') as file:
    #        content_base64 = base64.b64encode(content).decode('utf-8')
    #        content = file.read()
    #        save_to_github(content_base64, plot_filename, repo_info)
    #else:
    #    plt.savefig(os.path.join(folder_path, plot_filename))

if __name__ == "__main__":
    main()
