import argparse
from amateur_pt import MODEL_TYPES
from torch import nn
import torch


def initialize_weights(policy, init_type):
    for m in policy.modules():
        if isinstance(m, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif init_type == "he":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                raise ValueError(f"Unknown initialization type: {init_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        help="Type of algorithm policy to pre-train, i.e. PPO, SAC, TD3, A2C",
    )
    parser.add_argument(
        "-o",
        "--out",
        "--output",
        type=str,
        help="Output folder to save the randomly initialized policy state dicts inside 'policies'",
    )
    parser.add_argument(
        "--algo-kwargs",
        dest="kwargs",
        nargs=argparse.REMAINDER,
        help="User-defined keyword arguments to pass to the algorithm class.",
    )
    args = parser.parse_args()
    if args.kwargs is None:
        args.kwargs = dict()

    model = MODEL_TYPES[args.algo.upper()]("MlpPolicy", "CartPole-v1", **args.kwargs)

    policy = model.policy.to("cpu")

    for init_type in ["xavier", "orthogonal", "he"]:
        initialize_weights(policy, init_type)
        state_dict = policy.state_dict()
        out_path = (
            f"policies/{args.out}/{init_type}_{args.algo.lower()}_initialization.pt"
        )
        torch.save(state_dict, out_path)
        print(f"Randomly initialized policy state dict saved to {out_path}")
