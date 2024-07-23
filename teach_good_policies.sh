#!/bin/bash
set -e

python3 policies/CartPole/good.py --algo PPO
python3 policies/CartPole/good.py --algo SAC
python3 policies/CartPole/good.py --algo TD3
python3 policies/CartPole/good.py --algo A2C

python3 policies/CliffWalking/good.py --algo PPO
python3 policies/CliffWalking/good.py --algo SAC
python3 policies/CliffWalking/good.py --algo TD3
python3 policies/CliffWalking/good.py --algo A2C

python3 policies/HalfCheetah/good.py --algo PPO
python3 policies/HalfCheetah/good.py --algo SAC
python3 policies/HalfCheetah/good.py --algo TD3
python3 policies/HalfCheetah/good.py --algo A2C

python3 policies/Humanoid/good.py --algo PPO
python3 policies/Humanoid/good.py --algo SAC
python3 policies/Humanoid/good.py --algo TD3
python3 policies/Humanoid/good.py --algo A2C

python3 policies/LunarLander/good.py --algo PPO
python3 policies/LunarLander/good.py --algo SAC
python3 policies/LunarLander/good.py --algo TD3
python3 policies/LunarLander/good.py --algo A2C

python3 policies/MountainCar/good.py --algo PPO
python3 policies/MountainCar/good.py --algo SAC
python3 policies/MountainCar/good.py --algo TD3
python3 policies/MountainCar/good.py --algo A2C

python3 policies/Walker2D/good.py --algo PPO
python3 policies/Walker2D/good.py --algo SAC
python3 policies/Walker2D/good.py --algo TD3
python3 policies/Walker2D/good.py --algo A2C