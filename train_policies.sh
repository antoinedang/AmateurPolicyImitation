#!/bin/bash
set -e

policy_types=("xavier" "orthogonal" "he" "good" "bad" "pure_random")
algos=("ppo" "sac" "td3" "a2c")
env_configs=("Pendulum-v1,300_000,Pendulum" "BipedalWalker-v3,100_000,BipedalWalker" "LunarLanderContinuous-v2,300_000,LunarLander" "MountainCarContinuous-v0,300_000,MountainCar")
# TODO: "Walker2D-v4,1_000_000,Walker2D" "HalfCheetah-v4,1_000_000,HalfCheetah" "Humanoid-v4,2_000_000,Humanoid"

# For loop
for algo in "${algos[@]}"
do
    for policy_type in "${policy_types[@]}"
    do
        for tuple in "${env_configs[@]}"
        do
            IFS=',' read -r envid steps policy_folder <<< "$tuple"
            echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/"${policy_type}"_$algo.pt"
            python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/${policy_type}_$algo.pt
        done
    done
done