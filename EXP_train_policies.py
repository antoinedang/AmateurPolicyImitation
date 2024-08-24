import os
import sys

python_bin = "C:/Users/antoi/.pyenv/pyenv-win/versions/3.11.0/python.exe"

algos = ["ppo", "sac"]  # , "td3", "a2c"
policy_types = ["xavier", "orthogonal", "he", "good", "bad", "pure_random"]
env_configs = [
    "Pendulum-v1,500_000,Pendulum",
    "LunarLanderContinuous-v2,300_000,LunarLander",
    "Hopper-v4,1_000_000,Hopper",
    "Humanoid-v4,2_000_000,Humanoid",
]

starting_parameter_combo = ""  # inclusive
ending_parameter_combo = ""  # exclusive

normalize = "--normalize"  # make empty to disable normalization


def main():
    global starting_parameter_combo
    global ending_parameter_combo

    for algo in algos:
        for policy_type in policy_types:
            for env_config in env_configs:
                envid, steps, policy_folder = env_config.split(",")

                current_combo = f"{algo},{policy_type},{policy_folder}"

                if (
                    starting_parameter_combo != current_combo
                    and starting_parameter_combo != ""
                ):
                    continue
                starting_parameter_combo = ""

                if ending_parameter_combo == current_combo:
                    sys.exit(0)

                if policy_type in ["xavier", "orthogonal", "he"]:
                    init_file = f"policies/{policy_folder}/{policy_type}_{algo}_initialization.pt"
                else:
                    init_file = f"policies/{policy_folder}/{policy_type}_{algo}.pt"

                command = f"{python_bin} train.py --algo {algo} --env {envid} --n-steps {steps} --init {init_file} {normalize}"
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()
