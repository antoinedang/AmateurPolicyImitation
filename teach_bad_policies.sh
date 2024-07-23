#!/bin/bash
set -e

python3 policies/CartPole/bad.py --algo PPO
python3 policies/CartPole/bad.py --algo SAC
python3 policies/CartPole/bad.py --algo TD3
python3 policies/CartPole/bad.py --algo A2C

python3 policies/CliffWalking/bad.py --algo PPO
python3 policies/CliffWalking/bad.py --algo SAC
python3 policies/CliffWalking/bad.py --algo TD3
python3 policies/CliffWalking/bad.py --algo A2C

python3 policies/HalfCheetah/bad.py --algo PPO
python3 policies/HalfCheetah/bad.py --algo SAC
python3 policies/HalfCheetah/bad.py --algo TD3
python3 policies/HalfCheetah/bad.py --algo A2C

python3 policies/Humanoid/bad.py --algo PPO
python3 policies/Humanoid/bad.py --algo SAC
python3 policies/Humanoid/bad.py --algo TD3
python3 policies/Humanoid/bad.py --algo A2C

python3 policies/LunarLander/bad.py --algo PPO
python3 policies/LunarLander/bad.py --algo SAC
python3 policies/LunarLander/bad.py --algo TD3
python3 policies/LunarLander/bad.py --algo A2C

python3 policies/MountainCar/bad.py --algo PPO
python3 policies/MountainCar/bad.py --algo SAC
python3 policies/MountainCar/bad.py --algo TD3
python3 policies/MountainCar/bad.py --algo A2C

python3 policies/Walker2D/bad.py --algo PPO
python3 policies/Walker2D/bad.py --algo SAC
python3 policies/Walker2D/bad.py --algo TD3
python3 policies/Walker2D/bad.py --algo A2C