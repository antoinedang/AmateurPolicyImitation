#!/bin/bash
set -e

algos=("ppo" "sac" "td3" "a2c")
policy_types=("pure_random" "good" "bad")
envs=("MountainCar" "Pendulum" "BipedalWalker" "LunarLander") 
# TODO: "HalfCheetah" "Humanoid" "Walker2D"

starting_parameter_combo="ppo,pure_random,Pendulum" # inclusive
ending_parameter_combo="sac,pure_random,MountainCar" # exclusive

# For loop
for algo in "${algos[@]}"
do
    for policy_type in "${policy_types[@]}"
    do
        for env in "${envs[@]}"
        do

            if [ "$starting_parameter_combo" != "$algo,$policy_type,$env" ] && [ "$starting_parameter_combo" != "" ]; then
                starting_parameter_combo=""
                continue
            fi
            
            if [ "$ending_parameter_combo" == "$algo,$policy_type,$env" ]; then
                exit 0
            fi

            echo "python3 policies/$env/$policy_type.py --algo $algo"
            python3 policies/$env/$policy_type.py --algo $algo
        done
    done
done