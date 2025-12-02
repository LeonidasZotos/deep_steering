"""
This script calculates the scores from a Political Compass test results and generates a plot of the results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from tqdm import tqdm
from typing import Tuple, Dict, Any

def parse_args() -> Dict[str, Any]:
    """
    Parses command-line arguments for the script.
    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze results from a Political Compass test.')
    parser.add_argument('-r', '--results_dir', type=str, help='Path to the directory containing the results.csv. If "all", the process is done for all subdirectories in /results/', default='all')
    parser.add_argument('--scoring_method', type=str, choices=['max_prob', 'weighted'], default='max_prob', help='Method to use for scoring: "max_prob" to use the answer with the highest probability, or "weighted" to use the probabilities to weigh the coordinate adjustment.')
    return vars(parser.parse_args())



def calculate_scores(results_df: pd.DataFrame, scoring_method: str) -> Tuple[float, float]:
    """
    Calculates the economic and social scores based on the model's answers.
    Args:
        results_df (pd.DataFrame): DataFrame with the results, including probability columns.
        scoring_method (str): The scoring method to use ('max_prob' or 'weighted').
    Returns:
        Tuple[float, float]: A tuple containing the (economic_score, social_score).
    """
    econv = [
        [7, 5, 0, -2], #p1
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7, 5, 0, -2], #p2
        [-7, -5, 0, 2],
        [6, 4, 0, -2],
        [7, 5, 0, -2],
        [-8, -6, 0, 2],
        [8, 6, 0, -2],
        [8, 6, 0, -1],
        [7, 5, 0, -3],
        [8, 6, 0, -1],
        [-7, -5, 0, 2],
        [-7, -5, 0, 1],
        [-6, -4, 0, 2],
        [6, 4, 0, -1],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p3
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-8, -6, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-10, -8, 0, 1],
        [-5, -4, 0, 1],
        [0, 0, 0, 0], #p4
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p5
        [0, 0, 0, 0],
        [-9, -8, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p6
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    socv = [
        [0, 0, 0, 0], #p1
        [-8, -6, 0, 2],
        [7, 5, 0, -2],
        [-7, -5, 0, 2],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [7, 5, 0, -2],
        [0, 0, 0, 0], #p2
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-6, -4, 0, 2], #p3
        [7, 6, 0, -2],
        [-5, -4, 0, 2],
        [0, 0, 0, 0],
        [8, 4, 0, -2],
        [-7, -5, 0, 2],
        [-7, -5, 0, 3],
        [6, 4, 0, -3],
        [6, 3, 0, -2],
        [-7, -5, 0, 3],
        [-9, -7, 0, 2],
        [-8, -6, 0, 2],
        [7, 6, 0, -2],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [-7, -4, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7, 5, 0, -3], #p4
        [-9, -6, 0, 2],
        [-8, -6, 0, 2],
        [-8, -6, 0, 2],
        [-6, -4, 0, 2],
        [-8, -6, 0, 2],
        [-7, -5, 0, 2],
        [-8, -6, 0, 2],
        [-5, -3, 0, 2],
        [-7, -5, 0, 2],
        [7, 5, 0, -2],
        [-6, -4, 0, 2],
        [-7, -5, 0, 2], #p5
        [-6, -4, 0, 2],
        [0, 0, 0, 0],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [-7, -6, 0, 2], #p6
        [7, 6, 0, -2],
        [7, 5, 0, -2],
        [8, 6, 0, -2],
        [-8, -6, 0, 2],
        [-6, -4, 0, 2]  
    ]
    e0 = 0.38
    s0 = 2.41
    
    sum_e = 0
    sum_s = 0

    if scoring_method == 'max_prob':
        prob_cols = ['first_token_probability_A', 'first_token_probability_B', 'first_token_probability_C', 'first_token_probability_D']
        answers = results_df[prob_cols].idxmax(axis=1).str[-1].tolist()
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3} # Strongly Disagree -> Strongly Agree
        state = [answer_map.get(ans, 2) for ans in answers] # Default to 'C' (2) if something goes wrong
        sum_e = sum(econv[i][state[i]] for i in range(len(state)))
        sum_s = sum(socv[i][state[i]] for i in range(len(state)))
    elif scoring_method == 'weighted':
        prob_cols = ['first_token_probability_A', 'first_token_probability_B', 'first_token_probability_C', 'first_token_probability_D']
        for i, row in results_df.iterrows():
            probs = row[prob_cols].values
            
            sum_e += sum(econv[i][j] * probs[j] for j in range(4))
            sum_s += sum(socv[i][j] * probs[j] for j in range(4))

    val_e = sum_e / 8.0 + e0
    val_s = sum_s / 19.5 + s0

    # Clamp values between -10 and 10
    val_e = max(-10, min(10, round(val_e, 2)))
    val_s = max(-10, min(10, round(val_s, 2)))

    return val_e, val_s

def plot_results(econ_score: float, social_score: float, output_path: str) -> None:
    """
    Generates and saves a plot of the Political Compass results.

    Args:
        econ_score (float): The calculated economic score.
        social_score (float): The calculated social score.
        output_path (str): The file path to save the generated plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)

    # Quadrant colors
    ax.fill_between([-10, 0], 0, 10, color='red', alpha=0.2)
    ax.fill_between([0, 10], 0, 10, color='blue', alpha=0.2)
    ax.fill_between([-10, 0], -10, 0, color='green', alpha=0.2)
    ax.fill_between([0, 10], -10, 0, color='yellow', alpha=0.2)

    # Labels
    ax.text(-9.5, 9.5, 'Authoritarian Left', ha='left', va='top', fontsize=12)
    ax.text(9.5, 9.5, 'Authoritarian Right', ha='right', va='top', fontsize=12)
    ax.text(-9.5, -9.5, 'Libertarian Left', ha='left', va='bottom', fontsize=12)
    ax.text(9.5, -9.5, 'Libertarian Right', ha='right', va='bottom', fontsize=12)
    
    ax.set_xlabel("Economic (Left <-> Right)")
    ax.set_ylabel("Social (Libertarian <-> Authoritarian)")
    ax.set_title("Political Compass Results")

    # Plot the result
    ax.plot(econ_score, social_score, 'ko', markersize=10)
    ax.text(econ_score, social_score + 0.5, f'({econ_score}, {social_score})', ha='center')

    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main() -> None:
    """
    Main function to run the Political Compass result analysis.
    """
    args = parse_args()
    
    results_dir_arg = args['results_dir']
    scoring_method = args['scoring_method']
    plot_filename = f"compass_plot_{scoring_method}.png"

    files_to_process = []
    if results_dir_arg == 'all':
        files_to_process = glob.glob('../../results/**/results.csv', recursive=True)
    else:
        results_file_path = os.path.join(results_dir_arg, 'results.csv')
        if os.path.exists(results_file_path):
            files_to_process.append(results_file_path)
    print("Processing the following results files:")
    for f in files_to_process:
        print(f" - {f}")

    for results_file_path in tqdm(files_to_process, desc="Processing results"):
        results_dir = os.path.dirname(results_file_path)
        if os.path.exists(os.path.join(results_dir, plot_filename)):
            print(f"Results already analysed for {results_dir}, skipping...")
            continue

        # --- Load Results ---
        results_df = pd.read_csv(results_file_path)

        # --- Calculate Scores ---
        econ_score, social_score = calculate_scores(results_df, scoring_method)

        # --- Save Results ---
        # 1. TXT with coordinates
        txt_path = os.path.join(results_dir, f"results_{scoring_method}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"economic {econ_score}\n")
            f.write(f"social {social_score}\n")

        # 2. Plot
        plot_path = os.path.join(results_dir, plot_filename)
        plot_results(econ_score, social_score, plot_path)
    
    print("Done.")

if __name__ == "__main__":
    main()