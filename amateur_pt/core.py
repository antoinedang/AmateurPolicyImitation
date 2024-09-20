from typing import Union, Dict, Tuple, Any, Optional, Callable, Iterator
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from stable_baselines3 import *
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy as _evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import PolicyPredictor
from gymnasium import Env
from scipy.spatial import distance
from gymnasium.wrappers import ClipAction, NormalizeObservation


class AmateurTeacher:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def predict(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]], *_, **__
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        actions = []
        for obs in observation:
            actions.append(self.get_action(obs))
        return np.array(actions), None

    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.generate_observation(self.seed)
        # TODO: NORMALIZE OBSERVATION
        # NormalizeObservation class
        amateur_action = self.get_action(obs)

        try:
            amateur_action = self.clipper.action(amateur_action)
        except:
            self.clipper = ClipAction(gym.make(self.env_id))
            amateur_action = self.clipper.action(amateur_action)

        return obs, amateur_action

    def train(
        self,
        student: BaseAlgorithm,
        batch_size: int,
        epochs: int,
        teacher_interactions_per_epoch: int,
        make_optimizer: Callable[[Iterator[torch.nn.Parameter]], optim.Optimizer],
        make_scheduler: Callable[[optim.Optimizer], lr_scheduler.LRScheduler],
        log_interval: float = 100,
        device: str = "auto",
    ) -> Dict[str, Any]:
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {"num_workers": 1} if device == torch.device("cuda") else {}
        dataset = AmateurDataset(self, teacher_interactions_per_epoch)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )

        env = student.env
        if isinstance(env.action_space, gym.spaces.Box):
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        model = student.policy.to(device)

        def _train(model, device, train_loader, optimizer):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        action = model(data)
                    action_prediction = action.double()
                    target = target.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target_class = action_prediction * 0
                    target_class[target] = 1
                    target = target_class
                loss = criterion(action_prediction, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]    Loss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                        + "\t\t\t\t\t",
                        end="\r",
                    )

        # Define an Optimizer and a learning rate schedule.
        optimizer = make_optimizer(model.parameters())
        scheduler = make_scheduler(optimizer)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            _train(model, device, data_loader, optimizer)
            print()
            scheduler.step()

        return model.state_dict()

    def get_action(observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate_observation(self, seed: Optional[int] = None) -> np.ndarray:
        return self.observation_space.sample()


class AmateurDataset(Dataset):
    def __init__(
        self,
        teacher: AmateurTeacher,
        len: int,
    ):
        self.teacher = teacher
        self.len = len
        # INITIALIZE DATA HERE
        obs = np.array([self.teacher.generate_sample()[0] for _ in range(len)])
        # GET STD AND MEAN FOR NORMALIZATION PARAMS
        self.obs_mean = np.mean(obs, axis=0)
        self.obs_std = np.std(obs, axis=0)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.teacher.generate_sample()
        return torch.tensor((x - self.obs_mean) / self.obs_std), torch.tensor(
            y, dtype=torch.float32
        )


def evaluate_policy(
    env: Union[str, Env],
    policy: PolicyPredictor,
    teacher: Optional[AmateurTeacher] = None,
) -> Tuple[float, float, float]:

    if not isinstance(policy, AmateurTeacher) and teacher is None:
        raise ValueError(
            "Teacher must be provided when evaluating a non-amateur policy."
        )
    if isinstance(env, str):
        env = Monitor(gym.make(env))
    gen_obs_fn = (
        teacher.generate_observation
        if teacher is not None
        else policy.generate_observation
    )
    avg_reward, std_reward = _evaluate_policy(policy, env, render=False)

    actions = []
    for _ in range(100_000):
        obs = gen_obs_fn()
        action, _ = policy.predict([obs], None, None, None)
        actions.append(action[0])
    actions = np.array(actions)

    if isinstance(env.action_space, gym.spaces.Discrete):
        class_counts = np.bincount(actions.flatten())
        class_balance_score = (
            -1.0 * sum(np.abs(class_counts - np.mean(class_counts))) / len(actions)
        )
    elif isinstance(env.action_space, gym.spaces.Box):

        def average_distance_to_nearest_neighbor(points):
            num_points = points.shape[0]
            distances = np.zeros(num_points)
            for i in range(num_points):
                # Calculate distances from point i to all other points
                point_distances = distance.cdist(points[i : i + 1], points)[0]
                # Exclude distance to itself (which is 0)
                point_distances = np.delete(point_distances, i)
                # Find the minimum distance
                min_distance = np.min(point_distances)
                # Store the minimum distance
                distances[i] = min_distance

            # Calculate the average distance to nearest neighbor
            avg_distance = np.mean(distances)

            return avg_distance

        class_balance_score = average_distance_to_nearest_neighbor(actions)
    else:
        raise NotImplementedError(
            "Only Box and Discrete action spaces are supported. Got {}".format(
                env.action_space
            )
        )

    return avg_reward, std_reward, class_balance_score
