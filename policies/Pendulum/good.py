from amateur_pt import *
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import PendulumEnv
import math


class PendulumAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = PendulumEnv().observation_space
        self.env_id = "Pendulum-v1"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        angle = math.acos(observation[0])
        ang_vel = observation[2]
        if abs(angle) < 0.5:
            return np.array([-10 * ang_vel])
        else:
            if abs(ang_vel) > 4:
                return np.array([0])
            if ang_vel < 0:
                return np.array([-2.0])
            else:
                return np.array([2.0])

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    teacher = PendulumAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
