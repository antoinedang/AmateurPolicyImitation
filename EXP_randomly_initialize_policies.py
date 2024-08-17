import os
import sys

algos = ["ppo", "sac", "td3", "a2c"]
envs = ["HalfCheetah", "Humanoid", "Walker2D"]
starting_parameter_combo = ""  # inclusive
ending_parameter_combo = ""  # non-inclusive


def main():
    global starting_parameter_combo
    global ending_parameter_combo

    for algo in algos:
        for env in envs:
            current_combo = f"{algo},{env}"

            if (
                starting_parameter_combo != current_combo
                and starting_parameter_combo != ""
            ):
                starting_parameter_combo = ""
                continue

            if ending_parameter_combo == current_combo:
                sys.exit(0)

            command = (
                f"python3 randomly_initialize_policies.py --algo {algo} --env {env}"
            )
            print(command)
            os.system(command)


if __name__ == "__main__":
    main()
