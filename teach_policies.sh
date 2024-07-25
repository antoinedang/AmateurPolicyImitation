#!/bin/bash
set -e

policy_types=("pure_random" "good" "bad")
algos=("ppo" "sac" "td3" "a2c")
# DONE: "Pendulum" "BipedalWalker"
envs=("LunarLander" "MountainCar") 
# TODO: #"HalfCheetah" "Humanoid" "Walker2D"

# For loop
for algo in "${algos[@]}"
do
    for env in "${envs[@]}"
    do
        for policy_type in "${policy_types[@]}"
        do
            echo "python3 policies/$env/$policy_type.py --algo $algo"
            python3 policies/$env/$policy_type.py --algo $algo
        done
    done
done