import os
import sys

python_bin = "/usr/bin/python"

algos = ["ppo", "sac", "td3", "a2c"]
envs = ["Humanoid", "Hopper"]
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
                continue
            starting_parameter_combo = ""

            if ending_parameter_combo == current_combo:
                sys.exit(0)

            command = f"{python_bin} randomly_initialize_policies.py --algo {algo} --env {env}"
            print(command)
            os.system(command)


if __name__ == "__main__":
    main()
