from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
import numpy as np
from gymnasium.envs.mujoco import HumanoidEnv
import random


class HumanoidAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = HumanoidEnv().observation_space
        self.action_space = HumanoidEnv().action_space
        self.env_id = "Humanoid-v4"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        action = np.zeros(self.action_space.shape)
        torso_quat = observation[1:5]
        joint_angles = observation[5:22]
        
        TARGET_ANGLES = [
            0,
            -0.2,
            0,
            0,
            0,
            random.random() * 0.2,
            random.random() * 0.2,
            0,
            0,
            random.random() * 0.2,
            random.random() * 0.2,
            0.6,
            0.6,
            -0.5,
            0,
            0,
            -0.5,
        ]
        
        for i in range(len(joint_angles)):
            joint_angle = joint_angles[i]
            action[i] = -10 * (joint_angle - TARGET_ANGLES[i])

        return np.clip(action, self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    teacher = HumanoidAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
