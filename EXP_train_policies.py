import os
import sys

python_bin = "/usr/bin/python"

# algos = ["ppo", "sac", "td3", "a2c"]
algos = ["ppo", "sac"]
# policy_types = ["xavier", "orthogonal", "he", "good", "bad", "pure_random"]
policy_types = ["xavier", "orthogonal", "he", "good"]
env_configs = [
    "Pendulum-v1,500_000,Pendulum",
    # "LunarLanderContinuous-v2,300_000,LunarLander",
    # "Hopper-v4,1_000_000,Hopper",
    "Humanoid-v4,2_000_000,Humanoid",
]

starting_parameter_combo = ""  # inclusive
ending_parameter_combo = ""  # exclusive

normalization = "--normalize"


def main():
    global starting_parameter_combo
    global ending_parameter_combo

    for algo in algos:
        for env_config in env_configs:
            for policy_type in policy_types:
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

                command = f"{python_bin} train.py --algo {algo} --env {envid} --n-steps {steps} --init {init_file} {normalization}"
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()
