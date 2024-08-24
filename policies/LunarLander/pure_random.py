from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
import numpy as np
from gymnasium.envs.box2d import LunarLander


class LunarLanderAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = LunarLander(continuous=True).observation_space
        self.action_space = LunarLander(continuous=True).action_space
        self.env_id = "LunarLanderContinuous-v2"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


if __name__ == "__main__":
    teacher = LunarLanderAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
