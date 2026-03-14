"""
main.py
=======
Single entry point for all experiments in the assignment:
  - Standard training (MLP, CNN, VGG, ResNet, MobileNet)
  - Transfer learning from ImageNet pretrained models (Options 1 & 2)
  - Label smoothing
  - Knowledge distillation (standard and teacher_prob)

Usage examples:
  # Standard training
  python main.py --model resnet  --dataset cifar10 --epochs 20
  python main.py --model mlp     --dataset mnist   --epochs 10

  # Transfer learning — run both options and compare
  python main.py --mode transfer --model resnet18  --epochs 10
  python main.py --mode transfer --model vgg16     --epochs 10 --option 1

  # Label smoothing
  python main.py --model resnet --dataset cifar10 --label_smoothing 0.1

  # Knowledge distillation (train ResNet teacher first, then distil)
  python main.py --model cnn --dataset cifar10 --distillation --teacher_path best_resnet.pth
  python main.py --model mobilenet --dataset cifar10 --distillation --distill_mode teacher_prob
"""

import random
import ssl
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from parameters import TrainingConfig, get_params
from models.MLP       import MLP
from models.CNN       import MNIST_CNN, SimpleCNN
from models.VGG       import VGG
from models.ResNet    import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from pretrained       import (
    build_resnet18_option1, build_resnet18_option2,
    build_vgg16_option1,    build_vgg16_option2,
    run_transfer, TrainingHistory,
    plot_training_curves, plot_accuracy_bar, plot_tsne, print_results_table,
)
from train import run_training
from test  import run_test

ssl._create_default_https_context = ssl._create_unverified_context


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fix all randomness sources for reproducible training runs.

    Each library maintains its own random state; all must be seeded to
    guarantee identical results across runs with the same seed.
    cudnn.deterministic=True forces CUDA to select reproducible (though
    sometimes slower) kernel implementations.

    Args:
        seed: Any integer (e.g. 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(config: TrainingConfig) -> torch.device:
    """
    Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        config: TrainingConfig instance (device field used as a hint).

    Returns:
        torch.device: Selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: TrainingConfig) -> nn.Module:
    """
    Instantiate the model specified in config.model.

    Args:
        config: TrainingConfig instance.

    Returns:
        nn.Module: Untrained model ready to be moved to a device.

    Raises:
        ValueError: Unknown model name or incompatible dataset.
    """
    name: str = config.model
    nc:   int = config.num_classes

    if name == "mlp":
        return MLP(
            input_size   = config.input_size,
            hidden_sizes = config.hidden_sizes,
            num_classes  = nc,
            dropout      = config.dropout,
        )
    if name == "cnn":
        return MNIST_CNN(num_classes=nc) if config.dataset == "mnist" \
               else SimpleCNN(num_classes=nc)
    if name == "vgg":
        if config.dataset == "mnist":
            raise ValueError("VGG requires 3-channel input; use --dataset cifar10.")
        return VGG(dept=config.vgg_depth, num_class=nc)
    if name == "resnet":
        if config.dataset == "mnist":
            raise ValueError("ResNet requires 3-channel input; use --dataset cifar10.")
        return ResNet(BasicBlock, config.resnet_layers, num_classes=nc)
    if name == "mobilenet":
        if config.dataset == "mnist":
            raise ValueError("MobileNetV2 requires 3-channel input; use --dataset cifar10.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model '{name}'. Choose: mlp, cnn, vgg, resnet, mobilenet.")


def load_teacher(
    config: TrainingConfig,
    device: torch.device,
) -> Optional[nn.Module]:
    """
    Load the pre-trained ResNet-18 teacher for knowledge distillation.

    The teacher's parameters are frozen (requires_grad=False) so that:
      - No gradient graph is allocated for teacher tensors.
      - The optimiser receives only student parameters.
      - Accidental weight updates are impossible.

    Args:
        config: Must have distillation=True and a valid teacher_path.
        device: Compute device.

    Returns:
        Optional[nn.Module]: Loaded frozen teacher, or None if KD is off.
    """
    if not config.distillation:
        return None

    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=config.num_classes)
    state   = torch.load(config.teacher_path, map_location=device)
    teacher.load_state_dict(state)
    teacher.to(device).eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded from: {config.teacher_path}")
    return teacher


# ─────────────────────────────────────────────────────────────────────────────
# Transfer Learning Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_mode(config: TrainingConfig, device: torch.device) -> None:
    """
    Run transfer learning experiments and plot Option 1 vs Option 2 comparison.

    If config.transfer_option == 0, both options are run and compared.
    Otherwise only the specified option is executed.

    Calls the builder functions from pretrained.py, then run_transfer() for
    the actual training loop, and finally plot_transfer_comparison() to
    produce a bar chart saved to disk.

    Args:
        config: TrainingConfig instance (model, transfer_option, epochs, lr).
        device: Compute device.
    """
    options_to_run = [1, 2] if config.transfer_option == 0 else [config.transfer_option]
    from typing import List as _List
    histories: _List[TrainingHistory] = []
    models_trained = []

    for opt in options_to_run:
        if config.model == "resnet18":
            model = (build_resnet18_option1(freeze_backbone=config.freeze_backbone)
                     if opt == 1 else build_resnet18_option2())
        elif config.model == "vgg16":
            model = (build_vgg16_option1(freeze_backbone=config.freeze_backbone)
                     if opt == 1 else build_vgg16_option2())
        else:
            raise ValueError(
                f"Transfer learning supports 'resnet18' and 'vgg16', got '{config.model}'."
            )

        model = model.to(device)
        label = f"{config.model}_option{opt}"
        history = run_transfer(
            model      = model,
            option     = opt,
            model_name = label,
            epochs     = config.epochs,
            device     = device,
            lr         = config.learning_rate,
            batch_size = config.batch_size,
        )
        histories.append(history)
        models_trained.append((model, opt, label))

    print_results_table(histories)
    plot_training_curves(histories, save_prefix=config.model)
    plot_accuracy_bar(histories,    save_prefix=config.model)

    if config.tsne:
        for model, opt, label in models_trained:
            plot_tsne(model, opt, label, device, save_prefix=config.model)




# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Program entry point.

    Dispatches to the correct experiment based on config.mode:
      "transfer"         ->  Transfer learning (Option 1 and/or 2)
      "train" / "both"   ->  Standard training loop
      "test"  / "both"   ->  Evaluation on the test split
    """
    config = get_params()
    set_seed(config.seed)
    device = get_device(config)

    print(f"\n{'=' * 55}")
    print(f"  Dataset          : {config.dataset}")
    print(f"  Model            : {config.model}")
    print(f"  Mode             : {config.mode}")
    print(f"  Device           : {device}")
    if config.label_smoothing > 0:
        print(f"  Label smoothing  : epsilon={config.label_smoothing}")
    if config.distillation:
        print(f"  Distillation     : T={config.temperature}  "
              f"alpha={config.distill_alpha}  mode={config.distill_mode}")
    print(f"{'=' * 55}\n")

    # ── Transfer learning ────────────────────────────────────────────────────
    if config.mode == "transfer":
        run_transfer_mode(config, device)
        return

    # ── Standard training + evaluation ──────────────────────────────────────
    model   = build_model(config).to(device)
    teacher = load_teacher(config, device)

    if config.mode in ("train", "both"):
        run_training(model, config, device, teacher=teacher)

    if config.mode in ("test", "both"):
        run_test(model, config, device)


if __name__ == "__main__":
    main()