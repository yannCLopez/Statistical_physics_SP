import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse
import requests
import base64
import io

def get_github_files(repo_owner, repo_name, path=''):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    response = requests.get(url)
    if response.status_code == 200:
        contents = response.json()
        return [item['name'] for item in contents if item['type'] == 'file' and item['name'].startswith('P_K_log_spaced_') and item['name'].endswith('.txt')]
    else:
        print(f"Error accessing GitHub repository: {response.status_code}")
        return []

def get_local_files(directory):
    return [f for f in os.listdir(directory) if f.startswith('P_K_log_spaced_') and f.endswith('.txt')]

def download_github_file(repo_owner, repo_name, file_path):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.json()['content']
        decoded_content = base64.b64decode(content).decode('utf-8')
        return io.StringIO(decoded_content)
    else:
        print(f"Error downloading file from GitHub: {response.status_code}")
        return None

def parse_filename(filename):
    # Pattern for p and q distributions
    pattern = r'P_K_log_spaced_p([\d\.-]+)_q([\d\.-]+)_x([\d\.]+)'
    match = re.search(pattern, filename)
    
    if match:
        p_dist = match.group(1).replace('-', ',')
        q_dist = match.group(2).replace('-', ',')
        x_crit = float(match.group(3))
        
        # Parse p distribution
        p_values = re.findall(r'(\d+)-(\d+\.\d+)', p_dist)
        p_str = ", ".join([f"{v}:{p}" for v, p in p_values])
        
        # Parse q distribution
        q_values = re.findall(r'(\d+)-(\d+\.\d+)', q_dist)
        q_str = ", ".join([f"{v}:{p}" for v, p in q_values])
        
        return f"p: [{p_str}], q: [{q_str}], x_crit: {x_crit}"
    
    # Pattern for m and n
    pattern_mn = r'P_K_log_spaced_m(\d+)_n(\d+)_x([\d\.]+)'
    match_mn = re.search(pattern_mn, filename)
    
    if match_mn:
        m = int(match_mn.group(1))
        n = int(match_mn.group(2))
        x_crit = float(match_mn.group(3))
        return f"m: {m}, n: {n}, x_crit: {x_crit}"
    
    return "Unable to parse filename"

def display_file_options(files):
    print("Available data files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    #    print(f"   {parse_filename(file)}")
    print()

def get_user_choice(files):
    while True:
        try:
            choice = int(input("Enter the number of the file you want to plot: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_data(file_content):
    P_K_conditional_large_k = {}
    for line in file_content:
        k, p = line.strip().split('\t')
        P_K_conditional_large_k[int(k)] = float(p)
    return P_K_conditional_large_k

def load_data_from_file(file_path):
    P_K_conditional_large_k = {}
    with open(file_path, 'r') as file:
        for line in file:
            k, p = line.strip().split('\t')
            P_K_conditional_large_k[int(k)] = float(p)
    return P_K_conditional_large_k

def load_data_from_content(file_content):
    P_K_conditional_large_k = {}
    for line in file_content.splitlines():
        k, p = line.strip().split('\t')
        P_K_conditional_large_k[int(k)] = float(p)
    return P_K_conditional_large_k

def process_data(P_K_conditional_large_k):
    k_values_filtered = []
    P_K_values_filtered = []
    for k, p in P_K_conditional_large_k.items():
        if k > 10**3 and p > 0:
            k_values_filtered.append(float(k))
            P_K_values_filtered.append(float(p))
    return np.array(k_values_filtered), np.array(P_K_values_filtered)

def fit_power_law(k_values, P_K_values):
    log_k = np.log(k_values)
    log_P_K = np.log(P_K_values)
    slope, intercept, _, _, _ = linregress(log_k, log_P_K)
    b = -slope
    a = np.exp(intercept)
    return a, b
def plot_results(k_values, P_K_values, a, b, filename, save_path, is_github, repo_info=None):
    plt.figure(figsize=(10, 6))
    plt.loglog(k_values, P_K_values, marker='o', linestyle='-', color='b', label="P(K = k)")
    plt.loglog(k_values, a / (k_values**b), linestyle='--', color='r', label=f"Fitted Power Law: a/k^b")
    plt.xlabel("k")
    plt.ylabel("P(K = k)")
    plt.title(f"Log-Log Plot of P(K = k) with Power Law Fit (k > 10^3)\n{filename}")
    plt.legend()
    plt.grid(True)
    
    # Add annotations for the power-law parameters a and b on the bottom left corner of the plot
    plt.text(0.05, 0.05, f"a = {a:.4e}", transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.text(0.05, 0.01, f"b = {b:.4f}", transform=plt.gca().transAxes, fontsize=12, color='black')

    # Save the plot
    save_filename = "plot_" + os.path.splitext(filename)[0] + ".png"
    if is_github:
        # Save to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        # Upload to GitHub
        upload_to_github(img_data, save_filename, repo_info)
    else:
        save_full_path = os.path.join(save_path, save_filename)
        plt.savefig(save_full_path)
        print(f"Plot saved to: {save_full_path}")
    
    plt.show()

def upload_to_github(img_data, filename, repo_info):
    owner, repo, path, token = repo_info
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print(f"Attempting to upload to URL: {url}")  # Debug print
    
    # Check if file already exists
    response = requests.get(url, headers=headers)
    print(f"GET request status code: {response.status_code}")  # Debug print
    if response.status_code == 200:
        # File exists, get its SHA
        sha = response.json()["sha"]
    else:
        sha = None

    # Prepare the file content
    content = base64.b64encode(img_data.getvalue()).decode("utf-8")
    data = {
        "message": "Add/Update plot",
        "content": content,
    }
    if sha:
        data["sha"] = sha

    # Create or update the file
    response = requests.put(url, headers=headers, json=data)
    print(f"PUT request status code: {response.status_code}")  # Debug print
    if response.status_code in [201, 200]:
        print(f"Plot successfully uploaded to GitHub: {url}")
    else:
        print(f"Failed to upload plot to GitHub. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Request URL: {url}")  # Debug print
        print(f"Request headers: {headers}")  # Debug print (be careful not to expose the full token)
        print(f"Request data: {data}")  # D

def main():
    parser = argparse.ArgumentParser(description="Plot statistical physics data from GitHub or local directory")
    parser.add_argument("--repo", help="GitHub repository in the format owner/repo/folder")
    parser.add_argument("--local", help="Local directory path")
    parser.add_argument("--file", help="Specify the file to plot")
    parser.add_argument("--token", help="GitHub personal access token")
    args = parser.parse_args()

    if args.repo:
        if not args.token:
            print("Please provide a GitHub personal access token using the --token argument")
            return
        repo_parts = args.repo.split('/')
        if len(repo_parts) < 2:
            print("Please provide a valid GitHub repository in the format owner/repo or owner/repo/folder")
            return
        repo_owner = repo_parts[0]
        repo_name = repo_parts[1]
        folder_path = '/'.join(repo_parts[2:]) if len(repo_parts) > 2 else ''
        
        files = get_github_files(repo_owner, repo_name, folder_path)
        is_github = True
        repo_info = (repo_owner, repo_name, folder_path, args.token)
    elif args.local:
        if not os.path.isdir(args.local):
            print(f"The specified local directory does not exist: {args.local}")
            return
        files = get_local_files(args.local)
        is_github = False
        repo_info = None
    else:
        print("Please provide either a GitHub repository (--repo) or a local directory (--local)")
        return

    if not files:
        print("No suitable data files found in the specified location.")
        return

    if args.file:
        if args.file in files:
            chosen_file = args.file
        else:
            print(f"Specified file '{args.file}' not found. Please choose from the available files.")
            display_file_options(files)
            chosen_file = get_user_choice(files)
    else:
        display_file_options(files)
        chosen_file = get_user_choice(files)

    if is_github:
        file_path = f"{folder_path}/{chosen_file}" if folder_path else chosen_file
        file_content = download_github_file(repo_owner, repo_name, file_path)
        if file_content is None:
            return
        P_K_data = load_data_from_content(file_content.getvalue())
        save_path = None  # Not used for GitHub
    else:
        file_path = os.path.join(args.local, chosen_file)
        P_K_data = load_data_from_file(file_path)
        save_path = args.local

    k_values, P_K_values = process_data(P_K_data)
    a, b = fit_power_law(k_values, P_K_values)

    print(f"\nEstimated power-law parameters for {chosen_file}:")
    print(f"a = {a:.4e}")
    print(f"b = {b:.4f}")

    plot_results(k_values, P_K_values, a, b, chosen_file, save_path, is_github, repo_info)

if __name__ == "__main__":
    main()
