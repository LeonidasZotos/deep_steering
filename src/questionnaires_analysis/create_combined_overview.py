"""
This script combines scores from a Political Compass test results of a defined experiment and generates a single plot of the results.
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args() -> Dict[str, Any]:
    """
    Parses command-line arguments for the script.
    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Create a combined plot from a Political Compass test experiment.')
    parser.add_argument('-e', '--experiment_name', type=str, required=True, help='Name of the experiment to plot, as defined in the setups file.')
    parser.add_argument('--scoring_method', type=str, choices=['max_prob', 'weighted'], default='max_prob', help='Method to use for scoring: "max_prob" or "weighted".')
    parser.add_argument('--setups_file', type=str, default='experiments.json', help='Path to the JSON file with experiment setups.')
    return vars(parser.parse_args())


def read_scores(file_path: str) -> Tuple[float, float]:
    """
    Reads economic and social scores from a text file.

    Args:
        file_path (str): The path to the file containing the scores.

    Returns:
        Tuple[float, float]: A tuple containing the economic and social scores.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    econ_score = float(lines[0].strip().split()[1])
    social_score = float(lines[1].strip().split()[1])
    return econ_score, social_score


def plot_combined_results(results: List[Tuple[float, float, str]], output_path: str) -> None:
    """
    Generates and saves a plot of the combined Political Compass results.

    Args:
        results (List[Tuple[float, float, str]]): A list of tuples, where each tuple is (econ_score, social_score, description).
        output_path (str): The file path to save the generated plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)

    # Quadrant colors
    ax.fill_between([-10, 0], 0, 10, color='red', alpha=0.2)
    ax.fill_between([0, 10], 0, 10, color='blue', alpha=0.2)
    ax.fill_between([-10, 0], -10, 0, color='green', alpha=0.2)
    ax.fill_between([0, 10], -10, 0, color='purple', alpha=0.2)

    # Labels
    ax.text(-9.5, 9.5, 'Authoritarian Left', ha='left', va='top', fontsize=12)
    ax.text(9.5, 9.5, 'Authoritarian Right', ha='right', va='top', fontsize=12)
    ax.text(-9.5, -9.5, 'Libertarian Left', ha='left', va='bottom', fontsize=12)
    ax.text(9.5, -9.5, 'Libertarian Right', ha='right', va='bottom', fontsize=12)

    ax.set_xlabel("Economic (Left <-> Right)")
    ax.set_ylabel("Social (Libertarian <-> Authoritarian)")
    ax.set_title("Political Compass Results - Combined")

    # Plot the results
    for econ_score, social_score, description in results:
        ax.plot(econ_score, social_score, 'o', markersize=8)
        ax.text(econ_score, social_score + 0.3, f'{description}\n({econ_score:.2f}, {social_score:.2f})', ha='center', fontsize=8)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """
    Main function to run the Political Compass result combination and plotting.
    """
    args = parse_args()

    experiment_name = args['experiment_name']
    scoring_method = args['scoring_method']
    setups_file_path = args['setups_file']

    # --- Load Setups ---
    if not os.path.exists(setups_file_path):
        print(f"Error: Setups file not found at '{setups_file_path}'")
        return

    with open(setups_file_path, 'r') as f:
        setups = json.load(f)

    experiment_setup = setups.get(experiment_name)
    if not experiment_setup:
        print(f"Error: Experiment '{experiment_name}' not found in '{setups_file_path}'")
        return

    results_data = []
    # Assuming the script is in src/questionnaires_analysis, results are in ../../results
    base_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))

    for run_key, run_info in tqdm(experiment_setup.items(), desc=f"Processing experiment '{experiment_name}'"):
        experiment_id = run_info.get('experiment_id')
        description = run_info.get('description')

        if not experiment_id or not description:
            print(f"Warning: Skipping run '{run_key}' due to missing 'experiment_id' or 'description'.")
            continue

        # Find the result file by searching for the experiment_id directory
        search_pattern = os.path.join(base_results_dir, '*', experiment_id, f"results_{scoring_method}.txt")
        found_files = glob.glob(search_pattern)

        if not found_files:
            print(f"Warning: Could not find results file for run '{description}' (ID: {experiment_id}). Searched with pattern: {search_pattern}")
            continue

        if len(found_files) > 1:
            print(f"Warning: Found multiple results files for run '{description}' (ID: {experiment_id}). Using the first one: {found_files[0]}")

        results_file_path = found_files[0]

        try:
            econ_score, social_score = read_scores(results_file_path)
            results_data.append((econ_score, social_score, description))
        except Exception as e:
            print(f"Error reading or parsing scores for '{description}' (ID: {experiment_id}) from file {results_file_path}: {e}")

    if not results_data:
        print("No data points were collected. Cannot generate a plot.")
        return

    # --- Save Combined Results ---
    output_dir = os.path.join(base_results_dir, 'combined_plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{experiment_name}_{scoring_method}.png")

    plot_combined_results(results_data, plot_path)

    print(f"\nCombined plot saved to: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
