#!/bin/bash
set -e

python3 policies/CartPole/pure_random.py --algo PPO
python3 policies/CartPole/pure_random.py --algo SAC
python3 policies/CartPole/pure_random.py --algo TD3
python3 policies/CartPole/pure_random.py --algo A2C

python3 policies/CliffWalking/pure_random.py --algo PPO
python3 policies/CliffWalking/pure_random.py --algo SAC
python3 policies/CliffWalking/pure_random.py --algo TD3
python3 policies/CliffWalking/pure_random.py --algo A2C

python3 policies/HalfCheetah/pure_random.py --algo PPO
python3 policies/HalfCheetah/pure_random.py --algo SAC
python3 policies/HalfCheetah/pure_random.py --algo TD3
python3 policies/HalfCheetah/pure_random.py --algo A2C

python3 policies/Humanoid/pure_random.py --algo PPO
python3 policies/Humanoid/pure_random.py --algo SAC
python3 policies/Humanoid/pure_random.py --algo TD3
python3 policies/Humanoid/pure_random.py --algo A2C

python3 policies/LunarLander/pure_random.py --algo PPO
python3 policies/LunarLander/pure_random.py --algo SAC
python3 policies/LunarLander/pure_random.py --algo TD3
python3 policies/LunarLander/pure_random.py --algo A2C

python3 policies/MountainCar/pure_random.py --algo PPO
python3 policies/MountainCar/pure_random.py --algo SAC
python3 policies/MountainCar/pure_random.py --algo TD3
python3 policies/MountainCar/pure_random.py --algo A2C

python3 policies/Walker2D/pure_random.py --algo PPO
python3 policies/Walker2D/pure_random.py --algo SAC
python3 policies/Walker2D/pure_random.py --algo TD3
python3 policies/Walker2D/pure_random.py --algo A2C