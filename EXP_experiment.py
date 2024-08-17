import os
import subprocess


def main():
    # List of scripts to run
    scripts = [
        # "python3 EXP_randomly_initialize_policies.py",
        # "python3 EXP_teach_policies.py",
        "python3 EXP_train_policies.py",
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
