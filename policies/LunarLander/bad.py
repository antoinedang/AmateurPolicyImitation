from amateur_pt import AmateurTeacher, transfer_knowledge_and_save
from typing import Optional
from gymnasium.envs.box2d import LunarLander
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import random


class LunarLanderAmateurTeacher(AmateurTeacher):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.observation_space = LunarLander().observation_space
        self.env_id = "LunarLander-v2"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        x, y, vx, vy, angle, vangle, contact0, contact1 = observation

        if x < -0.05:
            # need to move right (lean slightly right)
            if angle > 0.4:
                return np.array([3])
            elif angle < 0:
                return np.array([1])
            else:
                if random.random() < 0.75:
                    return np.array([2])
                else:
                    return np.array([0])
        elif x > 0.05:
            # need to move right (lean slightly right)
            if angle > 0:
                return np.array([3])
            elif angle < -0.4:
                return np.array([1])
            else:
                if random.random() < 0.75:
                    return np.array([2])
                else:
                    return np.array([0])

        else:
            if angle < 0.1:
                return np.array([3])
            elif angle > -0.1:
                return np.array([1])
            else:
                return np.array([0])

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


if __name__ == "__main__":
    teacher = LunarLanderAmateurTeacher(seed=0)
    training_kwargs = dict(
        epochs=100,
        teacher_interactions_per_epoch=int(4e5),
        make_optimizer=lambda params: optim.Adam(params, lr=0.05),
        make_scheduler=lambda optimizer: lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=1.0
        ),
        log_interval=100,
        device="auto",
    )
    algo_kwargs = dict()

    ##############################################

    pre_trained_state_dict = transfer_knowledge_and_save(
        teacher, teacher.env_id, training_kwargs, algo_kwargs, __file__
    )
