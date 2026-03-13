"""
parameters.py
=============
Parses command-line arguments and returns a typed TrainingConfig dataclass.

Why dataclass instead of dict?
  - Access via config.learning_rate instead of params["learning_rate"].
  - Type hints are enforced; IDEs provide autocomplete and static analysis.
  - Accidental key typos are caught at definition time, not at runtime.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DataClass Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Holds all hyperparameters and settings for a training run.

    The @dataclass decorator auto-generates __init__, __repr__, and __eq__.
    Fields with mutable defaults (lists, tuples) must use field(default_factory=...)
    to avoid sharing the same object across all instances — a classic Python bug.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset:     str                = "mnist"   # "mnist" or "cifar10"
    data_dir:    str                = "./data"
    num_workers: int                = 2
    mean:        Tuple[float, ...]  = field(default_factory=tuple)
    std:         Tuple[float, ...]  = field(default_factory=tuple)

    # ── Model ─────────────────────────────────────────────────────────────────
    model:         str       = "mlp"
    input_size:    int       = 784
    hidden_sizes:  List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes:   int       = 10
    dropout:       float     = 0.3
    vgg_depth:     str       = "16"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

    # ── Transfer Learning ──────────────────────────────────────────────────────
    transfer_option: int  = 1      # 1 = resize + freeze backbone, 2 = modify conv layers
    freeze_backbone: bool = True   # if True, early layers are frozen in option 1

    # ── Label Smoothing ────────────────────────────────────────────────────────
    label_smoothing: float = 0.0   # 0.0 = disabled; typical value: 0.1

    # ── Knowledge Distillation ─────────────────────────────────────────────────
    distillation:  bool  = False
    teacher_path:  str   = "best_resnet.pth"
    temperature:   float = 4.0    # T > 1 softens the teacher's probability distribution
    distill_alpha: float = 0.3    # weight of the soft loss; (1 - alpha) weights the hard loss
    distill_mode:  str   = "standard"  # "standard" or "teacher_prob"

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:        int   = 10
    batch_size:    int   = 64
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:         int = 42
    device:       str = "cpu"
    save_path:    str = "best_model.pth"
    log_interval: int = 100
    mode:         str = "both"    # "train", "test", "both", or "transfer"


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def get_params() -> TrainingConfig:
    """
    Parses command-line arguments and returns a populated TrainingConfig.

    Returns:
        TrainingConfig: Fully populated configuration object.

    Examples:
        python main.py --model resnet --dataset cifar10 --epochs 20
        python main.py --model cnn --distillation --teacher_path best_resnet.pth
        python main.py --model resnet --dataset cifar10 --label_smoothing 0.1
        python main.py --model mobilenet --distillation --distill_mode teacher_prob
    """
    parser = argparse.ArgumentParser(
        description="CIFAR-10 / MNIST: Transfer Learning & Knowledge Distillation"
    )

    # ── Core arguments ────────────────────────────────────────────────────────
    parser.add_argument("--mode",       choices=["train", "test", "both", "transfer"], default="both")
    parser.add_argument("--dataset",    choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",
        choices=["mlp", "cnn", "vgg", "resnet", "mobilenet", "resnet18", "vgg16"], default="mlp")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--batch_size", type=int,   default=64)

    # ── Model-specific ────────────────────────────────────────────────────────
    parser.add_argument("--vgg_depth",
        choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4,
        default=[2, 2, 2, 2], metavar=("L1", "L2", "L3", "L4"),
        help="Number of blocks per ResNet stage (default: 2 2 2 2 = ResNet-18)")

    # ── Transfer Learning ──────────────────────────────────────────────────────
    parser.add_argument("--transfer_option", type=int, choices=[0, 1, 2], default=1,
        # 0 = run both options and compare
        help="1: resize images + freeze backbone  |  2: modify early conv layers")
    parser.add_argument("--no_freeze", action="store_true",
        help="In option 1, fine-tune all layers instead of freezing the backbone")

    # ── Label Smoothing ────────────────────────────────────────────────────────
    parser.add_argument("--label_smoothing", type=float, default=0.0,
        help="Label smoothing epsilon (0.0 = disabled; recommended: 0.1)")

    # ── Knowledge Distillation ─────────────────────────────────────────────────
    parser.add_argument("--distillation", action="store_true",
        help="Enable knowledge distillation training mode")
    parser.add_argument("--teacher_path", type=str, default="best_resnet.pth",
        help="Path to the saved teacher model weights")
    parser.add_argument("--temperature", type=float, default=4.0,
        help="Distillation temperature T (higher = softer probability distribution)")
    parser.add_argument("--distill_alpha", type=float, default=0.3,
        help="Weight of the soft KD loss; hard CE loss weight = (1 - alpha)")
    parser.add_argument("--distill_mode", choices=["standard", "teacher_prob"],
        default="standard",
        help="standard: classic KD  |  teacher_prob: use teacher confidence on true class only")

    args = parser.parse_args()

    # ── Dataset-dependent statistics ──────────────────────────────────────────
    # Mean and std are pre-computed from the training split of each dataset.
    # Normalisation: (pixel - mean) / std  ->  roughly N(0, 1) distribution.
    if args.dataset == "mnist":
        input_size: int                = 784
        mean:       Tuple[float, ...]  = (0.1307,)
        std:        Tuple[float, ...]  = (0.3081,)
    else:
        input_size = 3072
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return TrainingConfig(
        dataset         = args.dataset,
        mean            = mean,
        std             = std,
        input_size      = input_size,
        model           = args.model,
        vgg_depth       = args.vgg_depth,
        resnet_layers   = args.resnet_layers,
        transfer_option = args.transfer_option,
        freeze_backbone = not args.no_freeze,
        label_smoothing = args.label_smoothing,
        distillation    = args.distillation,
        teacher_path    = args.teacher_path,
        temperature     = args.temperature,
        distill_alpha   = args.distill_alpha,
        distill_mode    = args.distill_mode,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        device          = args.device,
        mode            = args.mode,
    )
