from amateur_pt import *
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv
from torch import optim
from torch.optim import lr_scheduler


class MountainCarAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = MountainCarEnv().observation_space
        self.action_space = MountainCarEnv().action_space
        self.env_id = "MountainCarContinuous-v0"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    teacher = MountainCarAmateurTeacher(seed=0)
    training_kwargs = dict(
        epochs=100,
        teacher_interactions_per_epoch=int(4e4),
        make_optimizer=lambda params: optim.Adam(params, lr=1.0),
        make_scheduler=lambda optimizer: lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.7
        ),
        log_interval=100,
        device="auto",
    )
    algo_kwargs = dict()

    ##############################################

    pre_trained_state_dict = transfer_knowledge_and_save(
        teacher, teacher.env_id, training_kwargs, algo_kwargs, __file__
    )
