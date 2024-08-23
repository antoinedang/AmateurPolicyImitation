# take arguments: algorithm, policy initialization method, environment, stopping criteria,
import os
import json
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecCheckNan
import argparse
import gymnasium as gym
from amateur_pt import *
import torch
import time
from model_normalization import normalize_weights

###########################
##   ARGUMENT  PARSING   ##
###########################

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--algo", type=str, help="Algorithm to use for training (e.g. PPO, SAC, etc.)"
)
argparser.add_argument(
    "--init",
    type=str,
    help="Path to checkpoint to continue training from (torch state dict file)",
)
argparser.add_argument(
    "--env",
    type=str,
    help="Environment ID to train on (e.g. Pendulum-v1)",
)
argparser.add_argument(
    "--n-steps",
    type=int,
    help="Total timesteps to train policy for, per randomization factor (can do less if reward threshold is reached early)",
)
argparser.add_argument(
    "--normalize",
    action="store_true",
    help="Pass flag to normalize model weights before training.",
)

args = argparser.parse_args()
print(args)

##########################
##  SETUP TRAIN PARAMS  ##
##########################

TOTAL_TIMESTEPS = args.n_steps
INITIAL_POLICY = args.init.lstrip().rstrip()
EVAL_FREQ = TOTAL_TIMESTEPS // 100
CHECKPOINT_FREQ = TOTAL_TIMESTEPS // 10
ENV_ID = args.env
MODEL = MODEL_TYPES[args.algo.upper()]

log_dir = os.path.splitext(INITIAL_POLICY)[0]
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

config_json_path = os.path.dirname(INITIAL_POLICY) + "/config.json"
config = json.load(open(config_json_path, "r"))
policy_kwargs = config[args.algo.upper()]["policy_kwargs"]
algo_kwargs = config[args.algo.upper()]["algo_kwargs"]

wall_clock_log_path = log_dir + "/total_training_seconds.txt"

##########################
##  ENVIRONMENT  SETUP  ##
##########################

env = VecMonitor(DummyVecEnv([lambda: gym.make(ENV_ID)]))
eval_env = VecMonitor(DummyVecEnv([lambda: gym.make(ENV_ID)]))

env = VecCheckNan(env, raise_exception=False)
eval_env = VecCheckNan(eval_env, raise_exception=False)

##########################
## MODEL INITIALIZATION ##
##########################

print("\nBeginning training.\n")

model = MODEL(
    policy="MlpPolicy",
    env=env,
    verbose=0,
    policy_kwargs=policy_kwargs,
    **algo_kwargs,
)

state_dict = torch.load(INITIAL_POLICY)
if args.normalize:
    state_dict = normalize_weights(state_dict)

model.policy.load_state_dict(state_dict)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=0,
)

training_start_time = time.time()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback],
    log_interval=1,
    reset_num_timesteps=True,
    progress_bar=True,
)

total_training_time = time.time() - training_start_time

with open(wall_clock_log_path, "w+") as f:
    f.write(str(total_training_time) + " total training time (seconds)\n")
    f.write(str(TOTAL_TIMESTEPS / total_training_time) + " steps/second (average)\n")
