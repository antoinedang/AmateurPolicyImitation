from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
import numpy as np
from gymnasium.envs.mujoco import HumanoidEnv


class HumanoidAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = HumanoidEnv().observation_space
        self.action_space = HumanoidEnv().action_space
        self.env_id = "Humanoid-v4"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(np.ones(self.action_space.shape) * -10, self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    teacher = HumanoidAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
