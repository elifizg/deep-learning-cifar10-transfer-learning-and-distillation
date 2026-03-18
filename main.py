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
from train import run_training, run_training_tracked, plot_comparison_curves, print_comparison_table, RunHistory
from test  import run_test, count_flops

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
# Part B: Individual experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def _free_gpu(*models) -> None:
    """Move models to CPU and release GPU memory cache."""
    for m in models:
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


def _make_cifar_config(epochs: int, save_path: str,
                       label_smoothing: float = 0.0,
                       distillation: bool = False,
                       distill_mode: str = "standard",
                       temperature: float = 4.0,
                       distill_alpha: float = 0.3) -> TrainingConfig:
    """
    Build a CIFAR-10 TrainingConfig from scratch with explicit values.

    Avoids dataclasses.asdict() which converts tuples to lists and
    corrupts mean/std fields causing the config to silently fall back
    to MNIST defaults.
    """
    return TrainingConfig(
        dataset         = "cifar10",
        data_dir        = "./data",
        num_workers     = 0,
        mean            = (0.4914, 0.4822, 0.4465),
        std             = (0.2023, 0.1994, 0.2010),
        model           = "cnn",        # overridden per experiment
        input_size      = 3072,
        num_classes     = 10,
        dropout         = 0.3,
        vgg_depth       = "16",
        resnet_layers   = [2, 2, 2, 2],
        transfer_option = 1,
        freeze_backbone = True,
        label_smoothing = label_smoothing,
        distillation    = distillation,
        teacher_path    = "best_resnet.pth",
        temperature     = temperature,
        distill_alpha   = distill_alpha,
        distill_mode    = distill_mode,
        epochs          = epochs,
        batch_size      = 64,
        learning_rate   = 1e-3,
        weight_decay    = 1e-4,
        seed            = 42,
        device          = "cuda",
        save_path       = save_path,
        log_interval    = 100,
        mode            = "train",
        tsne            = False,
    )


def run_b1(config: TrainingConfig, device: torch.device) -> None:
    """B.1 — SimpleCNN baseline (standard CE, no teacher)."""
    from models.CNN import SimpleCNN
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_cnn_baseline.pth")
    print(f"  Dataset : cifar10 | Model : SimpleCNN | Device : {device}")
    model = SimpleCNN(num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="SimpleCNN (baseline)")
    _free_gpu(model)


def run_b2a(config: TrainingConfig, device: torch.device) -> None:
    """B.2a — ResNet-18 from scratch, no label smoothing."""
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_resnet.pth",
                             label_smoothing=0.0)
    print(f"  Dataset : cifar10 | Model : ResNet-18 | Device : {device}")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="ResNet (no LS)")
    _free_gpu(model)


def run_b2b(config: TrainingConfig, device: torch.device) -> None:
    """B.2b — ResNet-18 from scratch, label smoothing epsilon=0.1."""
    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_resnet_ls.pth",
                             label_smoothing=0.1)
    print(f"  Dataset : cifar10 | Model : ResNet-18 | Label Smoothing : 0.1 | Device : {device}")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="ResNet (LS=0.1)")
    _free_gpu(model)


def run_b3(config: TrainingConfig, device: torch.device) -> None:
    """B.3 — SimpleCNN student + ResNet teacher (standard KD)."""
    import os
    from models.CNN import SimpleCNN
    teacher_path = "best_resnet.pth"
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher weights not found: '{teacher_path}'\n"
            "  Run B.2a first to train and save the ResNet teacher:\n"
            "      python main.py --mode b2a --epochs 20"
        )
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_cnn_kd.pth",
                             distillation=True, distill_mode="standard",
                             temperature=4.0, distill_alpha=0.3)
    print(f"  Dataset : cifar10 | Model : SimpleCNN | KD T=4.0 alpha=0.3 | Device : {device}")
    model = SimpleCNN(num_classes=10).to(device)
    run_training_tracked(model, cfg, device, label="SimpleCNN (KD)", teacher=teacher)
    _free_gpu(model, teacher)


def run_b4(config: TrainingConfig, device: torch.device) -> None:
    """B.4 — MobileNet student + ResNet teacher (hybrid teacher_prob KD)."""
    import os
    from models.mobilenet import MobileNetV2
    teacher_path = "best_resnet.pth"
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher weights not found: '{teacher_path}'\n"
            "  Run B.2a first to train and save the ResNet teacher:\n"
            "      python main.py --mode b2a --epochs 20"
        )
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cfg = _make_cifar_config(epochs=config.epochs, save_path="best_mobilenet_kd.pth",
                             distillation=True, distill_mode="teacher_prob",
                             temperature=4.0, distill_alpha=0.3)
    print(f"  Dataset : cifar10 | Model : MobileNetV2 | Hybrid KD+LS | Device : {device}")
    model = MobileNetV2(num_classes=10).to(device)
    run_training_tracked(model, cfg, device,
                         label="MobileNet (hybrid KD+LS)", teacher=teacher)
    _free_gpu(model, teacher)


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Knowledge Distillation Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_kd_experiments(config: TrainingConfig, device: torch.device) -> None:
    """
    Run all Part B experiments in sequence and produce comparison plots.

    Delegates to individual run_b* functions so GPU is freed between each
    experiment.  Histories are collected and used to generate report plots.

    Args:
        config: TrainingConfig — only config.epochs is used.
        device: Compute device.
    """
    import json, os
    from train import RunHistory

    histories: list = []

    # Run each experiment — GPU freed after each one via _free_gpu()
    for mode, runner in [("b1", run_b1), ("b2a", run_b2a),
                         ("b2b", run_b2b), ("b3", run_b3), ("b4", run_b4)]:
        print(f"\n{'=' * 60}")
        print(f"  Running experiment: {mode.upper()}")
        print(f"{'=' * 60}")
        runner(config, device)

    # ── Collect histories from saved .pth files for plotting ─────────────────
    # Since each run_b* is independent, we re-run training in tracked mode
    # or load saved results. Here we rebuild histories from a quick eval.
    print("\n  All experiments complete. Generating plots...")

    # Auto-generate all comparison plots
    import importlib.util, os
    spec     = importlib.util.spec_from_file_location(
        "plot_results",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_results.py")
    )
    plot_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_mod)
    plot_mod.main()


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

    # For kd/b* modes the run_b* functions print their own headers.
    kd_modes = {"kd", "b1", "b2a", "b2b", "b3", "b4"}
    if config.mode not in kd_modes:
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
    else:
        print(f"\n  Mode : {config.mode}  |  Device : {device}\n")

    # ── Knowledge Distillation — individual experiments ──────────────────────
    kd_dispatch = {"b1": run_b1, "b2a": run_b2a, "b2b": run_b2b,
                   "b3": run_b3, "b4": run_b4}
    if config.mode in kd_dispatch:
        kd_dispatch[config.mode](config, device)
        return

    # ── Knowledge Distillation full pipeline ─────────────────────────────────
    if config.mode == "kd":
        run_kd_experiments(config, device)
        return

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