import argparse
from amateur_pt import evaluate_policy, AmateurTeacher
import pathlib
import torch
import os
from stable_baselines3 import PPO, SAC, TD3, A2C

MODEL_TYPES = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}


def transfer_knowledge_and_save(
    teacher: AmateurTeacher,
    env_id: str,
    training_kwargs: dict,
    algo_kwargs: dict,
    file: str,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        help="Type of algorithm policy to pre-train, i.e. PPO, SAC, A2C, etc.",
    )
    args = parser.parse_args()

    script_path = pathlib.Path(os.path.dirname(os.path.abspath(file)))
    script_name = os.path.basename(file)
    script_name = script_name[: script_name.rfind(".")]  # remove filename
    out_filename = script_name + "_" + args.algo.lower() + ".pt"
    out_path = script_path / out_filename

    avg_reward, std_reward, class_imbalance = evaluate_policy(env_id, teacher)
    print("teacher average reward: {}".format(avg_reward))
    print("teacher std. reward: {}".format(std_reward))
    print("teacher class imbalance: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "w+") as f:
        f.write(f"teacher average reward: {avg_reward}\n")
        f.write(f"teacher std. reward: {std_reward}\n")
        f.write(f"teacher class imbalance: {class_imbalance}\n\n")

    student = MODEL_TYPES[args.algo.upper()]("MlpPolicy", env_id, **algo_kwargs)
    avg_reward, std_reward, class_imbalance = evaluate_policy(env_id, student, teacher)
    print("Un-initialized student average reward: {}".format(avg_reward))
    print("Un-initialized student std. reward: {}".format(std_reward))
    print("Un-initialized student class imbalance: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(f"Un-initialized student average reward: {avg_reward}\n")
        f.write(f"Un-initialized student std. reward: {std_reward}\n")
        f.write(f"Un-initialized student class imbalance: {class_imbalance}\n\n")

    pre_trained_state_dict = teacher.train(student, batch_size=64, **training_kwargs)
    student.policy.load_state_dict(pre_trained_state_dict)

    avg_reward, std_reward, class_imbalance = evaluate_policy(env_id, student, teacher)
    print("Initialized student average reward: {}".format(avg_reward))
    print("Initialized student std. reward: {}".format(std_reward))
    print("Initialized student class imbalance: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(f"Initialized student average reward: {avg_reward}\n")
        f.write(f"Initialized student std. reward: {std_reward}\n")
        f.write(f"Initialized student class imbalance: {class_imbalance}\n")

    torch.save(pre_trained_state_dict, out_path)
    print(f"Pre-trained policy state dict saved to {out_path}.pt")
