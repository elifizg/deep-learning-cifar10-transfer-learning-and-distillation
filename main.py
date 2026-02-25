import random
import ssl
import numpy as np
import torch

from parameters import get_params
from models.MLP import MLP
from train import run_training
from test  import run_test


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):
    return MLP(
        input_size   = params["input_size"],
        hidden_sizes = params["hidden_sizes"],
        num_classes  = params["num_classes"],
        dropout      = params["dropout"],
    )


def main():
    params = get_params()

    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")

    device = torch.device(
        params["device"] if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params["mode"] in ("train", "both"):
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device)


if __name__ == "__main__":
    main()