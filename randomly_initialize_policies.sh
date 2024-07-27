#!/bin/bash
set -e

algos=("ppo" "sac" "td3" "a2c")
envs=("Pendulum" "BipedalWalker" "LunarLander" "MountainCar")
# TODO: #"HalfCheetah" "Humanoid" "Walker2D"

# For loop
for algo in "${algos[@]}"
do
    for env in "${envs[@]}"
    do
        echo "python3 randomly_initialize_policies.py --algo $algo --env $env"
        python3 randomly_initialize_policies.py --algo $algo --env $env
    done
done