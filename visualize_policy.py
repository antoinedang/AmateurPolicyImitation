import importlib
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

MODEL_TYPE = SAC
POLICY_NAME = "BipedalWalker"
POLICY_TYPE = "bad"  # bad, good, pure_random

USE_POLICY_CLASS = True
ENV_ID = "MountainCarContinuous-v0"
CKPT = "policies/MountainCar/good_ppo.pt"

if USE_POLICY_CLASS:
    policy_class_ = getattr(
        importlib.import_module("policies." + POLICY_NAME + "." + POLICY_TYPE),
        POLICY_NAME + "AmateurTeacher",
    )

    agent = policy_class_(seed=0)
    ENV_ID = agent.env_id
else:
    agent = MODEL_TYPE.load(
        path=CKPT,
        env=ENV_ID,
    )

env = Monitor(gym.make(ENV_ID, render_mode="human"))

while True:
    done = False
    total_reward = 0
    episode_length = 0
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()
    try:
        while not done:
            action, _ = agent.predict([obs], deterministic=True)
            action = action[0]
            try:
                obs, reward, done, _, _ = env.step(action)
            except:
                obs, reward, done, _ = env.step(action)
            if not done:
                episode_length += 1
                total_reward += reward
                # print(reward)
            env.render()
    except KeyboardInterrupt:
        print(
            " >>> Episode Length {}, Total Reward {}".format(
                episode_length, total_reward
            )
        )
