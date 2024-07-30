#!/bin/bash
set -e

envs=("MountainCar" "Pendulum" "BipedalWalker" "LunarLander" "HalfCheetah" "Humanoid" "Walker2D") 
algos=("ppo" "sac" "td3" "a2c")

for env in "${envs[@]}"
    do
    for algo in "${algos[@]}"
        do
        python3 plot_evaluations.py policies/${env}/good_${algo}/evaluations.npz policies/${env}/pure_random_${algo}/evaluations.npz policies/${env}/bad_${algo}/evaluations.npz policies/${env}/he_${algo}_initialization/evaluations.npz policies/${env}/xavier_${algo}_initialization/evaluations.npz policies/${env}/orthogonal_${algo}_initialization/evaluations.npz --names Good Random Bad He Xavier Orthogonal --save plot_${env}_${algo}.png
    done
done
