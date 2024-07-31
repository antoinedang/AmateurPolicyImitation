import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
import os


def generate_distinct_colors(num_colors):
    # Generate distinct colors in the HSV space
    hsv_colors = [(x / num_colors, 1.0, 1.0) for x in range(num_colors)]
    # Convert HSV colors to RGB
    rgb_colors = list(map(lambda x: mcolors.hsv_to_rgb(x), hsv_colors))
    return rgb_colors


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "eval_files", nargs="+", type=str, help="Path to .npz evaluation files."
)
argparser.add_argument(
    "--names", nargs="+", type=str, help="Names of evaluations (for plot labels)."
)
argparser.add_argument("--save", type=str, help="Filename to save plot to (as PNG)")
args = argparser.parse_args()

#####################

plt.figure(figsize=(10, 5))

num_curves = 0

for eval_file, name, color in zip(
    args.eval_files, args.names, generate_distinct_colors(len(args.eval_files))
):
    avg_rewards = []
    timesteps = []
    std_rewards = []

    try:
        evaluations = np.load(eval_file)
        num_curves += 1
    except FileNotFoundError:
        print(f"WARN: File {eval_file} not found.")
        continue
    timesteps = evaluations["timesteps"]
    avg_rewards = np.mean(evaluations["results"], axis=1)
    std_rewards = np.std(evaluations["results"], axis=1)

    # Plot the curve
    plt.plot(timesteps, avg_rewards, label=name, color=color)
    plt.fill_between(
        timesteps,
        avg_rewards - std_rewards,
        avg_rewards + std_rewards,
        color=color,
        alpha=0.2,
    )

if num_curves == 0:
    print("Exiting: No valid evaluation files found.")
    exit(0)

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

out_path = "plots/" + args.save

if os.path.exists(out_path):
    print(f"ERROR: File {out_path} already exists. Will not overwrite.")
    exit(0)
plt.savefig(out_path)
