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

# Define the distributions p and q

p = {2: mp.mpf('0.5'), 3: mp.mpf('0.5')}  # Example distribution for m
q = {2: mp.mpf('0.5'), 3: mp.mpf('0.5')}  # Example distribution for n
#p = {1: mp.mpf('0.1'), 2: mp.mpf('0.3'), 3: mp.mpf('0.6')}
#q = {1: mp.mpf('0.4'), 3: mp.mpf('0.3'), 5: mp.mpf('0.3')}
#p = {1: mp.mpf('0.25'), 2: mp.mpf('0.25'), 3: mp.mpf('0.5')}
#q = {2: mp.mpf('0.7'), 4: mp.mpf('0.3')}

def generate_generating_function(distribution):
    """
    Generate the generating function for a given distribution.
    """
    def G(x):
        return sum(prob * x**n for n, prob in distribution.items())
    return G

# Create generating functions
G_p = generate_generating_function(p)
G_q = generate_generating_function(q)

def compute_rho_precise(x, tol=mp.mpf('1e-12'), max_iter=1000000):
    """
    Compute rho(x) using high precision, iterating F(k) until convergence.
    """
    F_prev = mp.mpf(1.0)  # F(0) = 1 with high precision
    for _ in range(max_iter):
        F_next = G_p(1 - G_q(1 - x * F_prev))
        if abs(F_next - F_prev) < tol:
            return F_next
        F_prev = F_next
    return F_prev

def compute_P_K_precise_log_spaced(x, k_max, base=1.1, tol=mp.mpf('1e-12')):
    """
    Compute the probability distribution P(K = k) for logarithmically spaced k values.
    """
    # Step 1: Compute rho(x) with high precision
    rho = compute_rho_precise(x, tol=tol)

    # Step 2: Generate logarithmically spaced k values
    k_values = [0] + [int(base**i) for i in range(int(mp.log(k_max) / mp.log(base)) + 1)]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort

    # Step 3: Compute F(k) for the selected k values
    F = {0: mp.mpf(1.0)}  # F(0) = 1
    for k in k_values[1:]:
        Fk = G_p(1 - G_q(1 - x * F[k_values[k_values.index(k)-1]])) #k_values[k_values.index(k)-1] gets the previous k value in our logarithmically spaced list.
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
    P_K[k_values[-1]] = F[k_values[-1]] - F[k_values[-1]]

    return P_K

def chi(r):
    """
    Compute chi(r) using the new formula and numerical inversion.
    """
    r = mp.mpf(r)  # Ensure r is a mpmath float
    
    def inverse_G_p(y):
        return mp.findroot(lambda x: G_p(x) - y, 0.5)
    
    def inverse_G_q(y):
        return mp.findroot(lambda x: G_q(x) - y, 0.5)
    
    try:
        result = (1 - inverse_G_q(1 - inverse_G_p(r))) / r
        return float(result)  # Convert to Python float for compatibility with scipy (check)
    except (ValueError, ZeroDivisionError):
        return float('inf')  # Return infinity for invalid inputs

def compute_critical_values(precision=50):
    mp.dps = precision  # Set decimal precision
    
    def chi_wrapper(r):
        return chi(mp.mpf(r))
    
    # Use SciPy's minimize_scalar to find the global minimum
    result = minimize_scalar(chi_wrapper, bounds=(1e-10, 1), method='bounded') #By passing chi_wrapper without parentheses, we're giving minimize_scalar the ability to call the function with whatever arguments it needs during the optimization process.
    
    r_crit = mp.mpf(result.x)
    x_crit = mp.mpf(chi(r_crit))
    
    return r_crit, x_crit

def mpf_to_str(value):
    """Convert mpf object to string representation."""
    return str(value)

def get_distribution_string(dist):
    """Convert distribution dictionary to string representation."""
    return "_".join([f"{k}-{mpf_to_str(v)}" for k, v in dist.items()])

def save_metadata(filename, p, q, x):
    """Save metadata to a JSON file, handling mpf objects."""
    metadata = {
        "p_distribution": {str(k): mpf_to_str(v) for k, v in p.items()},
        "q_distribution": {str(k): mpf_to_str(v) for k, v in q.items()},
        "x_value": mpf_to_str(x)
    }
    metadata_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

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
        return

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
        print(f"Response: {response.text}")
    else:
        print(f"Failed to upload file to GitHub. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Request data: {data}")  # Be careful not to log sensitive information

    return response.status_code in [201, 200]

def main():
    parser = argparse.ArgumentParser(description="Compute and save P(K) distribution")
    parser.add_argument("--github", help="GitHub repository in the format owner/repo/folder")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument("--local", help="Local directory path")
    args = parser.parse_args()

    # Early check for required arguments
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
        save_function = save_to_github
    else:
        folder_path = args.local
        save_function = lambda content, filename, _: open(os.path.join(folder_path, filename), 'w').write(content)


    # Compute critical values
    r_crit, x_crit = compute_critical_values()

    print(f"r_crit = {mp.nstr(r_crit, 12)}")
    print(f"x_crit = {mp.nstr(x_crit, 12)}")

    # Parameters for high-precision computation
    x = x_crit  # High precision value for x
    k_max = 10**6  # Compute P(K=k) up to k=10^6
    base = 1.05  # Base for logarithmic spacing

    # Compute P(K = k) with high precision and logarithmic spacing
    P_K_precise_log_spaced = compute_P_K_precise_log_spaced(x, k_max, base)

    # Normalize the distribution to condition on finite K (ignore P(K = infinity))
    total_finite_sum = sum(P_K_precise_log_spaced.values())

    # Normalize P(K = k) by the total finite sum
    P_K_conditional_log_spaced = {k: P_K_precise_log_spaced[k] / total_finite_sum for k in P_K_precise_log_spaced}

    # Prepare the content to save
    p_string = get_distribution_string(p)
    q_string = get_distribution_string(q)
    filename = f"P_K_log_spaced_p{p_string}_q{q_string}_x{mpf_to_str(x)}.txt"
    
    content = ""
    for k, prob in P_K_conditional_log_spaced.items():
        content += f"{k}\t{mpf_to_str(prob)}\n"

    # Save the main file
    save_function(content, filename, repo_info if args.github else None)

    # Save the metadata
    metadata = {
        "p_distribution": {str(k): mpf_to_str(v) for k, v in p.items()},
        "q_distribution": {str(k): mpf_to_str(v) for k, v in q.items()},
        "x_value": mpf_to_str(x)
    }
    metadata_filename = filename.rsplit('.', 1)[0] + '_metadata.json'
    metadata_content = json.dumps(metadata, indent=2)
    save_function(metadata_content, metadata_filename, repo_info if args.github else None)

    print(f"Files saved: {filename} and {metadata_filename}")

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

    # Create and save the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(k_values_filtered, P_K_values_filtered, marker='o', linestyle='', color='b', label="P(K = k)")
    plt.loglog(k_values_filtered, a / (k_values_filtered**b), linestyle='--', color='r', label=f"Fitted Power Law: a/k^b")
    plt.xlabel("k")
    plt.ylabel("P(K = k)")
    plt.title("Log-Log Plot of P(K = k) with Power Law Fit (k > 10^3)")
    plt.legend()
    plt.grid(True)
    
    #plot_filename = filename.rsplit('.', 1)[0] + '_plot.png'
    #plt.savefig(plot_filename)
    
    #if args.github:
    #with open(plot_filename, 'rb') as file:
    #        content = file.read()
    #        content_base64 = base64.b64encode(content).decode('utf-8')
    #        save_to_github(content_base64, plot_filename, repo_info)
    #else:
    #    plt.savefig(os.path.join(folder_path, plot_filename))

    #print(f"Plot saved: {plot_filename}")

if __name__ == "__main__":
    main()
