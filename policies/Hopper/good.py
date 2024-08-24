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
        height = observation[0]
        top_angle, thigh_angle, leg_angle, foot_angle = [(180 * o) / np.pi for o in observation[1:5]]
        # print(f"top_angle: {top_angle}, thigh_angle: {thigh_angle}, leg_angle: {leg_angle}, foot_angle: {foot_angle}")
        # print(height)
        STRENGTH = 0.5
        knee_torque = 0.0
        thigh_torque = 0.0
        foot_torque = 2 * STRENGTH
        
        if leg_angle < -30 or height < 1.2:
            foot_torque = 2 * -STRENGTH
        
        if top_angle > 8:
            thigh_torque =-STRENGTH
            knee_torque = thigh_torque
        if top_angle < -8:
            thigh_torque = STRENGTH
            knee_torque = thigh_torque
            
        return np.clip(np.array([thigh_torque, knee_torque, foot_torque]), self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    teacher = HopperAmateurTeacher(seed=0)

    ##############################################

    transfer_knowledge_and_save(teacher, __file__)
