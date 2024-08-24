from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
import numpy as np
from gymnasium.envs.mujoco import HopperEnv


class HopperAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = HopperEnv().observation_space
        self.action_space = HopperEnv().action_space
        self.env_id = "Hopper-v4"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        height, top_angle, thigh_angle, leg_angle, foot_angle = observation


if __name__ == "__main__":
    teacher = HopperAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
