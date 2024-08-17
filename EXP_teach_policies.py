import os
import sys

algos = ["ppo", "sac", "td3", "a2c"]
policy_types = ["pure_random", "good", "bad"]
envs = ["MountainCar", "Pendulum", "BipedalWalker", "LunarLander"]
# TODO: "HalfCheetah" "Humanoid" "Walker2D"

starting_parameter_combo = "ppo,pure_random,Pendulum"  # inclusive
ending_parameter_combo = "sac,pure_random,MountainCar"  # exclusive


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
                    starting_parameter_combo = ""
                    continue

                if ending_parameter_combo == current_combo:
                    sys.exit(0)

                command = f"python3 policies/{env}/{policy_type}.py --algo {algo}"
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()
