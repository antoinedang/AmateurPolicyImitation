#!/bin/bash
set -e

python3 randomly_initialize_policies.py --algo PPO -o CartPole
python3 randomly_initialize_policies.py --algo SAC -o CartPole
python3 randomly_initialize_policies.py --algo TD3 -o CartPole
python3 randomly_initialize_policies.py --algo A2C -o CartPole

python3 randomly_initialize_policies.py --algo PPO -o CliffWalking
python3 randomly_initialize_policies.py --algo SAC -o CliffWalking
python3 randomly_initialize_policies.py --algo TD3 -o CliffWalking
python3 randomly_initialize_policies.py --algo A2C -o CliffWalking

python3 randomly_initialize_policies.py --algo PPO -o HalfCheetah
python3 randomly_initialize_policies.py --algo SAC -o HalfCheetah
python3 randomly_initialize_policies.py --algo TD3 -o HalfCheetah
python3 randomly_initialize_policies.py --algo A2C -o HalfCheetah

python3 randomly_initialize_policies.py --algo PPO -o Humanoid
python3 randomly_initialize_policies.py --algo SAC -o Humanoid
python3 randomly_initialize_policies.py --algo TD3 -o Humanoid
python3 randomly_initialize_policies.py --algo A2C -o Humanoid

python3 randomly_initialize_policies.py --algo PPO -o LunarLander
python3 randomly_initialize_policies.py --algo SAC -o LunarLander
python3 randomly_initialize_policies.py --algo TD3 -o LunarLander
python3 randomly_initialize_policies.py --algo A2C -o LunarLander

python3 randomly_initialize_policies.py --algo PPO -o MountainCar
python3 randomly_initialize_policies.py --algo SAC -o MountainCar
python3 randomly_initialize_policies.py --algo TD3 -o MountainCar
python3 randomly_initialize_policies.py --algo A2C -o MountainCar

python3 randomly_initialize_policies.py --algo PPO -o Walker2D
python3 randomly_initialize_policies.py --algo SAC -o Walker2D
python3 randomly_initialize_policies.py --algo TD3 -o Walker2D
python3 randomly_initialize_policies.py --algo A2C -o Walker2D