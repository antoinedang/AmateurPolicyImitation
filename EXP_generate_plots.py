import os

python_bin = "C:/Users/antoi/.pyenv/pyenv-win/versions/3.11.0/python.exe"

envs = [
    "MountainCar",
    "Pendulum",
    "BipedalWalker",
    "LunarLander",
    "HalfCheetah",
    "Humanoid",
    "Walker2D",
]
algos = ["ppo", "sac", "td3", "a2c"]


def main():
    for env in envs:
        for algo in algos:
            command = (
                f"{python_bin} plot_evaluations.py "
                f"policies/{env}/good_{algo}/evaluations.npz "
                f"policies/{env}/pure_random_{algo}/evaluations.npz "
                f"policies/{env}/bad_{algo}/evaluations.npz "
                f"policies/{env}/he_{algo}_initialization/evaluations.npz "
                f"policies/{env}/xavier_{algo}_initialization/evaluations.npz "
                f"policies/{env}/orthogonal_{algo}_initialization/evaluations.npz "
                f"--names Good Random Bad He Xavier Orthogonal "
                f"--save plot_{env}_{algo}.png"
            )
            print(command)
            os.system(command)


if __name__ == "__main__":
    main()
