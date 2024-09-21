import argparse
from amateur_pt.core import evaluate_policy, AmateurTeacher
import pathlib
import torch
import os
from stable_baselines3 import PPO, SAC, TD3, A2C
import time
from torch import optim
from torch.optim import lr_scheduler

MODEL_TYPES = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}


TRAINING_KWARGS = dict(
    epochs=100,
    teacher_interactions_per_epoch=int(4e5),
    make_optimizer=lambda params: optim.Adam(params, lr=0.001, weight_decay=1e-4),
    make_scheduler=lambda optimizer: lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=1.0
    ),
    log_interval=100,
    device="auto",
)
ALGO_KWARGS = dict()


def transfer_knowledge_and_save(
    teacher: AmateurTeacher,
    file: str,
):
    env_id = teacher.env_id
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
    print("Teacher average reward: {}".format(avg_reward))
    print("Teacher std. reward: {}".format(std_reward))
    print("Teacher class balance score: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "w+") as f:
        f.write(f"Teacher average reward: {avg_reward}\n")
        f.write(f"Teacher std. reward: {std_reward}\n")
        f.write(f"Teacher class balance score: {class_imbalance}\n\n")

    student = MODEL_TYPES[args.algo.upper()]("MlpPolicy", env_id, **ALGO_KWARGS)
    avg_reward, std_reward, class_imbalance = evaluate_policy(
        env_id, student, teacher, normalize=True
    )
    print("Un-initialized student average reward: {}".format(avg_reward))
    print("Un-initialized student std. reward: {}".format(std_reward))
    print("Un-initialized student class balance score: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(f"Un-initialized student average reward: {avg_reward}\n")
        f.write(f"Un-initialized student std. reward: {std_reward}\n")
        f.write(f"Un-initialized student class balance score: {class_imbalance}\n\n")

    start_teaching_time = time.time()
    pre_trained_state_dict = teacher.train(student, batch_size=64, **TRAINING_KWARGS)
    total_teaching_time = time.time() - start_teaching_time
    student.policy.load_state_dict(pre_trained_state_dict)

    avg_reward, std_reward, class_imbalance = evaluate_policy(
        env_id, student, teacher, normalize=True
    )
    print("Initialized student average reward: {}".format(avg_reward))
    print("Initialized student std. reward: {}".format(std_reward))
    print("Initialized student class balance score: {}".format(class_imbalance))
    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(f"Initialized student average reward: {avg_reward}\n")
        f.write(f"Initialized student std. reward: {std_reward}\n")
        f.write(f"Initialized student class balance score: {class_imbalance}\n\n")

    torch.save(pre_trained_state_dict, out_path)
    print(f"Pre-trained policy state dict saved to {out_path}.pt")

    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(f"Total time teaching: {total_teaching_time} seconds")
