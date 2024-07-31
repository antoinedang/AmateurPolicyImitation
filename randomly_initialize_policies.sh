#!/bin/bash
set -e

algos=("ppo" "sac" "td3" "a2c")
# "Pendulum" "BipedalWalker" "LunarLander" "MountainCar"
envs=("HalfCheetah" "Humanoid" "Walker2D")
starting_parameter_combo="" # inclusive
ending_parameter_combo="" # non-inclusive

# For loop
for algo in "${algos[@]}"
do
    for env in "${envs[@]}"
    do
        if [ "$starting_parameter_combo" != "$algo,$env" ] && [ "$starting_parameter_combo" != "" ]; then
            starting_parameter_combo=""
            continue
        fi
        
        if [ "$ending_parameter_combo" == "$algo,$env" ]; then
            exit 0
        fi

        echo "python3 randomly_initialize_policies.py --algo $algo --env $env"
        python3 randomly_initialize_policies.py --algo $algo --env $env
    done
done