import os
import sys

python_bin = "C:/Users/antoi/.pyenv/pyenv-win/versions/3.11.0/python.exe"

algos = ["ppo", "sac"]  # , "td3", "a2c"
policy_types = ["pure_random", "good", "bad"]
envs = [
    "Pendulum",
    "LunarLander",
    "Humanoid",
    "Hopper",
]

starting_parameter_combo = ""  # inclusive
ending_parameter_combo = ""  # exclusive


def main():
    global starting_parameter_combo
    global ending_parameter_combo

    for algo in algos:
        for policy_type in policy_types:
            for env in envs:
                current_combo = f"{algo},{policy_type},{env}"

                if (
                    starting_parameter_combo != f"{algo},{policy_type},{env}"
                    and starting_parameter_combo != ""
                ):
                    continue
                starting_parameter_combo = ""

                if ending_parameter_combo == current_combo:
                    sys.exit(0)

                command = f"{python_bin} policies/{env}/{policy_type}.py --algo {algo}"
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()
