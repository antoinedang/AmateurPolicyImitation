from amateur_pt import AmateurTrainer, MODEL_TYPES, evaluate_policy
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv
from stable_baselines3 import *
import argparse
import pickle
import os
import pathlib
from torch import optim
from torch.optim import lr_scheduler


class MountainCarAmateurTrainer(AmateurTrainer):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = MountainCarEnv().observation_space

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if abs(observation[1]) < 0.01:
            return np.array([1])
        return np.array([1]) if observation[1] > 0 else np.array([-1])

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        help="Type of algorithm policy to pre-train, i.e. PPO, SAC, A2C, etc.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Policy type to pre-train, i.e. MlpPolicy, CnnPolicy, etc.",
    )
    parser.add_argument(
        "-o",
        "--out",
        "--output",
        "--output_file",
        type=str,
        help="Output file to save the pre-trained policy state dict.",
    )
    args = parser.parse_args()

    trainer = MountainCarAmateurTrainer(seed=0)
    env_id = "MountainCarContinuous-v0"
    algo = MODEL_TYPES[args.algo.upper()]
    ppo_student = algo(args.policy, env_id)

    avg_reward, std_reward, class_imbalance = evaluate_policy(env_id, trainer)
    print("Trainer average reward: {}".format(avg_reward))
    print("Trainer std. reward: {}".format(std_reward))
    print("Trainer class imbalance: {}".format(class_imbalance))

    avg_reward, std_reward, class_imbalance = evaluate_policy(
        env_id, ppo_student, trainer
    )
    print("Un-initialized student average reward: {}".format(avg_reward))
    print("Un-initialized student std. reward: {}".format(std_reward))
    print("Un-initialized student class imbalance: {}".format(class_imbalance))

    pre_trained_state_dict = trainer.train(
        ppo_student,
        batch_size=64,
        epochs=2,
        expert_interactions_per_epoch=int(4e4),
        make_optimizer=lambda params: optim.Adam(params, lr=1.0),
        make_scheduler=lambda optimizer: lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.7
        ),
        log_interval=100,
        device="auto",
    )

    if args.out is not None:
        script_path = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        out_path = script_path / args.out
        with open(out_path, "wb") as f:
            pickle.dump(pre_trained_state_dict, f)
        print(f"Pre-trained policy state dict saved to {out_path}")

    ppo_student.policy.load_state_dict(pre_trained_state_dict)

    avg_reward, std_reward, class_imbalance = evaluate_policy(
        env_id, ppo_student, trainer
    )
    print("Initialized student average reward: {}".format(avg_reward))
    print("Initialized student std. reward: {}".format(std_reward))
    print("Initialized student class imbalance: {}".format(class_imbalance))
