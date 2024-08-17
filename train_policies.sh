#!/bin/bash
set -e

algos=("ppo" "sac" "td3" "a2c")
policy_types=("xavier" "orthogonal" "he" "good" "bad" "pure_random")
env_configs=("Pendulum-v1,300_000,Pendulum" "BipedalWalker-v3,500_000,BipedalWalker" "LunarLanderContinuous-v2,300_000,LunarLander" "MountainCarContinuous-v0,300_000,MountainCar")
# TODO: "Walker2D-v4,1_000_000,Walker2D" "HalfCheetah-v4,1_000_000,HalfCheetah" "Humanoid-v4,2_000_000,Humanoid"

starting_parameter_combo="ppo,good,Pendulum" # inclusive
ending_parameter_combo="sac,xavier,Pendulum" # exclusive

normalize="--normalize" # make empty to disable normalization

# For loop
for algo in "${algos[@]}"
do
    for policy_type in "${policy_types[@]}"
    do
        for tuple in "${env_configs[@]}"
        do
            IFS=',' read -r envid steps policy_folder <<< "$tuple"

            if [ "$starting_parameter_combo" != "$algo,$policy_type,$policy_folder" ] && [ "$starting_parameter_combo" != "" ]; then
                starting_parameter_combo=""
                continue
            fi
            
            if [ "$ending_parameter_combo" == "$algo,$policy_type,$policy_folder" ]; then
                exit 0
            fi

            starting_parameter_combo=""

            if [ "$policy_type" == "xavier" ] || [ "$policy_type" == "orthogonal" ] || [ "$policy_type" == "he" ]; then
                echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/"${policy_type}"_"${algo}"_initialization.pt $normalize"
                python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/${policy_type}_${algo}_initialization.pt $normalize
            else
                echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/"${policy_type}"_$algo.pt $normalize"
                python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/${policy_type}_$algo.pt $normalize
            fi
        done
    done
done