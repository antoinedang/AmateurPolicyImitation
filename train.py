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
    default=10,
    help="Path to checkpoint to continue training from (must point to .zip file, without the .zip extension in the path)",
)
argparser.add_argument(
    "--env",
    type=str,
    default=None,
    help="Environment ID to train on (e.g. CartPole-v1)",
)
argparser.add_argument(
    "--n-steps",
    type=int,
    default=None,
    help="Total timesteps to train policy for, per randomization factor (can do less if reward threshold is reached early)",
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

##########################
##  ENVIRONMENT  SETUP  ##
##########################

env = VecMonitor(DummyVecEnv([lambda: gym.make(ENV_ID)]))
eval_env = VecMonitor(DummyVecEnv([lambda: gym.make(ENV_ID)]))

env = VecCheckNan(env, raise_exception=True)
eval_env = VecCheckNan(eval_env, raise_exception=True)

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

model.policy.load_state_dict(torch.load(INITIAL_POLICY))

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=log_dir,
    name_prefix="ckpt",
    verbose=0,
)

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

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    log_interval=1,
    reset_num_timesteps=True,
    progress_bar=True,
)
