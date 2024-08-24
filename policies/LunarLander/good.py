from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
from gymnasium.envs.box2d import LunarLander
import numpy as np
import random


class LunarLanderAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = LunarLander(continuous=True).observation_space
        self.env_id = "LunarLanderContinuous-v2"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        x, y, vx, vy, angle, vangle, contact0, contact1 = observation

        if x > 0.05:
            # need to move right (lean slightly right)
            if angle > 0.4:
                return np.array([0, 1])
            elif angle < 0:
                return np.array([0, -1])
            else:
                return np.array([0.75, 0])
        elif x < -0.05:
            # need to move right (lean slightly right)
            if angle > 0:
                return np.array([0, 1])
            elif angle < -0.4:
                return np.array([0, -1])
            else:
                return np.array([0.75, 0])

        else:
            if angle > 0.1:
                return np.array([0, 1])
            elif angle < -0.1:
                return np.array([0, -1])
            else:
                return np.array([0, 0])


if __name__ == "__main__":
    teacher = LunarLanderAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
