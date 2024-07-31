from amateur_pt import *
from typing import Optional
import numpy as np
import random
from gymnasium.envs.classic_control import PendulumEnv


class PendulumAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = PendulumEnv().observation_space
        self.action_space = PendulumEnv().action_space
        self.env_id = "Pendulum-v1"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    teacher = PendulumAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
