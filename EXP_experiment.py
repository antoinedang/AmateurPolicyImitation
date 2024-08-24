import os
import subprocess

python_bin = "/usr/bin/python"


def main():
    # List of scripts to run
    scripts = [
        f"{python_bin} EXP_randomly_initialize_policies.py",
        f"{python_bin} EXP_teach_policies.py",
        f"{python_bin} EXP_train_policies.py",
    ]

    for script in scripts:
        print(f"Running {script}...")
        try:
            subprocess.run(script, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}: {e}")
            break


if __name__ == "__main__":
    main()
