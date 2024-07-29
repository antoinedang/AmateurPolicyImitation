#!/bin/bash
set -e

policy_types=("xavier" "orthogonal" "he" "good" "bad" "pure_random")
algos=("ppo" "sac" "td3" "a2c")
env_configs=("Pendulum-v1,300_000,Pendulum" "BipedalWalker-v3,500_000,BipedalWalker" "LunarLanderContinuous-v2,300_000,LunarLander" "MountainCarContinuous-v0,300_000,MountainCar")
# TODO: "Walker2D-v4,1_000_000,Walker2D" "HalfCheetah-v4,1_000_000,HalfCheetah" "Humanoid-v4,2_000_000,Humanoid"

starting_parameter_combo="good,sac,Pendulum"

# For loop
for algo in "${algos[@]}"
do
    for policy_type in "${policy_types[@]}"
    do
        for tuple in "${env_configs[@]}"
        do
            IFS=',' read -r envid steps policy_folder <<< "$tuple"

            if [ starting_parameter_combo != "$policy_type,$algo,$policy_folder" ] && [ starting_parameter_combo != "" ]; then
                continue
            fi

            starting_parameter_combo=""

            if [ "$policy_type" == "xavier" ] || [ "$policy_type" == "orthogonal" ] || [ "$policy_type" == "he" ]; then
                echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/"${policy_type}"_"${algo}"_initialization.pt"
                python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/${policy_type}_${algo}_initialization.pt
            else
                echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/"${policy_type}"_$algo.pt"
                python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/${policy_type}_$algo.pt
            fi
        done
    done
done