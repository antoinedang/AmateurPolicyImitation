from amateur_pt import *
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import (
    Continuous_MountainCarEnv,
)


class MountainCarAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = Continuous_MountainCarEnv().observation_space
        self.env_id = "MountainCarContinuous-v0"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if abs(observation[1]) < 0.01:
            return np.array([1])
        return np.array([1]) if observation[1] > 0 else np.array([-1])

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    teacher = MountainCarAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
