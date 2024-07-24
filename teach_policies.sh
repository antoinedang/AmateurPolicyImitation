#!/bin/bash
set -e

policy_type="pure_random"
algos=("ppo" "sac" "td3" "a2c")
envs=("CartPole" "CliffWalking" "HalfCheetah" "Humanoid" "LunarLander" "MountainCar" "Walker2D")

# For loop
for algo in "${algos[@]}"
do
    for env in "${envs[@]}"
    do
        echo "python3 policies/$env/$policy_type.py --algo $algo"
        python3 policies/$env/$policy_type.py --algo $algo
    done
done