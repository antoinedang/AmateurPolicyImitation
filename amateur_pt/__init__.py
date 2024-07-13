from amateur_pt.amateur_pt import AmateurTrainer, evaluate_policy
from stable_baselines3 import PPO, SAC, A2C, TD3, DQN, DDPG

MODEL_TYPES = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "TD3": TD3,
    "DQN": DQN,
    "DDPG": DDPG,
}
