#!/bin/bash
set -e

python3 policies/MountainCar/pure_random.py --algo ppo
python3 policies/MountainCar/good.py --algo ppo
python3 policies/MountainCar/bad.py --algo ppo

policy_types=("pure_random" "good" "bad")
# "ppo"
algos=("sac" "td3" "a2c")
envs=("MountainCar" "Pendulum" "BipedalWalker" "LunarLander") 
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