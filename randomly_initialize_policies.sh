#!/bin/bash
set -e

algos=("ppo" "sac" "td3" "a2c")
envs=("Pendulum" "BipedalWalker" "HalfCheetah" "Humanoid" "LunarLander" "MountainCar" "Walker2D")

# For loop
for algo in "${algos[@]}"
do
    for env in "${envs[@]}"
    do
        echo "python3 randomly_initialize_policies.py --algo $algo -o $env"
        python3 randomly_initialize_policies.py --algo $algo -o $env
    done
done