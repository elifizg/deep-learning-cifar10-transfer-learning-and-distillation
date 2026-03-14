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
    from models.CNN import SimpleCNN
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load("best_resnet.pth", map_location=device))
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
    from models.mobilenet import MobileNetV2
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load("best_resnet.pth", map_location=device))
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
    Run all Part B Knowledge Distillation experiments in sequence and
    produce comparison tables and plots.

    Experiment order (matches assignment requirements):
      B.1  SimpleCNN baseline             — standard CE, no teacher
      B.2a ResNet baseline                — standard CE, from scratch
      B.2b ResNet + Label Smoothing       — LS epsilon=0.1, from scratch
      B.3  SimpleCNN + KD                 — ResNet teacher, standard KD
      B.4  MobileNet + Hybrid KD+LS       — ResNet teacher, teacher_prob mode

    After all runs the function prints a combined results table with FLOPs
    and saves training curve PNG files for the report.

    Note: B.2a ResNet must finish before B.3 / B.4 because its saved weights
    are loaded as the teacher model.

    Args:
        config: Base TrainingConfig.  save_path and model fields are
                overridden per experiment inside this function.
        device: Compute device.
    """
    import copy as _copy
    from models.CNN      import SimpleCNN
    from models.ResNet   import ResNet, BasicBlock
    from models.mobilenet import MobileNetV2

    histories: list = []
    flops_dict: dict = {}

    def _cfg(**overrides) -> TrainingConfig:
        """Return a shallow copy of config with the given fields overridden."""
        import dataclasses
        d = dataclasses.asdict(config)
        d.update(overrides)
        return TrainingConfig(**d)

    # ── B.1  SimpleCNN baseline ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  B.1  SimpleCNN  —  standard CE")
    print("=" * 60)
    cfg_b1   = _cfg(model="cnn", dataset="cifar10",
                    save_path="best_cnn_baseline.pth",
                    distillation=False, label_smoothing=0.0)
    model_b1 = SimpleCNN(num_classes=10).to(device)
    h_b1     = run_training_tracked(model_b1, cfg_b1, device, label="SimpleCNN (baseline)")
    histories.append(h_b1)
    model_b1.cpu(); torch.cuda.empty_cache()

    # Measure FLOPs for SimpleCNN
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model_b1, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        flops_dict["SimpleCNN (baseline)"] = macs
    except Exception:
        flops_dict["SimpleCNN (baseline)"] = "N/A"

    # ── B.2a  ResNet baseline ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  B.2a  ResNet-18  —  no label smoothing")
    print("=" * 60)
    cfg_b2a   = _cfg(model="resnet", dataset="cifar10",
                     save_path="best_resnet.pth",
                     distillation=False, label_smoothing=0.0)
    model_b2a = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    h_b2a     = run_training_tracked(model_b2a, cfg_b2a, device, label="ResNet (no LS)")
    histories.append(h_b2a)
    model_b2a.cpu(); torch.cuda.empty_cache()

    try:
        macs, _ = get_model_complexity_info(
            model_b2a, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        flops_dict["ResNet (no LS)"] = macs
    except Exception:
        flops_dict["ResNet (no LS)"] = "N/A"

    # ── B.2b  ResNet + Label Smoothing ────────────────────────────────────────
    print()
    print("=" * 60)
    print("  B.2b  ResNet-18  —  label smoothing epsilon=0.1")
    print("=" * 60)
    cfg_b2b   = _cfg(model="resnet", dataset="cifar10",
                     save_path="best_resnet_ls.pth",
                     distillation=False, label_smoothing=0.1)
    model_b2b = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    h_b2b     = run_training_tracked(model_b2b, cfg_b2b, device, label="ResNet (LS=0.1)")
    histories.append(h_b2b)
    model_b2b.cpu(); torch.cuda.empty_cache()

    # ── B.3  SimpleCNN + KD  (ResNet teacher) ─────────────────────────────────
    print()
    print("=" * 60)
    print("  B.3  SimpleCNN  —  KD with ResNet teacher")
    print("=" * 60)
    # Load the best ResNet (B.2a) as teacher — frozen, eval mode
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    teacher.load_state_dict(torch.load("best_resnet.pth", map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cfg_b3   = _cfg(model="cnn", dataset="cifar10",
                    save_path="best_cnn_kd.pth",
                    distillation=True, distill_mode="standard",
                    temperature=4.0, distill_alpha=0.3, label_smoothing=0.0)
    model_b3 = SimpleCNN(num_classes=10).to(device)
    h_b3     = run_training_tracked(model_b3, cfg_b3, device,
                                    label="SimpleCNN (KD)", teacher=teacher)
    histories.append(h_b3)
    model_b3.cpu(); teacher.cpu(); torch.cuda.empty_cache()
    flops_dict["SimpleCNN (KD)"] = flops_dict.get("SimpleCNN (baseline)", "N/A")

    # ── B.4  MobileNet + Hybrid KD+LS ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("  B.4  MobileNet  —  Hybrid KD + LS (teacher_prob)")
    print("=" * 60)
    cfg_b4   = _cfg(model="mobilenet", dataset="cifar10",
                    save_path="best_mobilenet_kd.pth",
                    distillation=True, distill_mode="teacher_prob",
                    temperature=4.0, distill_alpha=0.3, label_smoothing=0.0)
    model_b4 = MobileNetV2(num_classes=10).to(device)
    h_b4     = run_training_tracked(model_b4, cfg_b4, device,
                                    label="MobileNet (hybrid KD+LS)", teacher=teacher)
    histories.append(h_b4)

    try:
        macs, _ = get_model_complexity_info(
            model_b4, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        flops_dict["MobileNet (hybrid KD+LS)"] = macs
    except Exception:
        flops_dict["MobileNet (hybrid KD+LS)"] = "N/A"

    # ── Final Reports ──────────────────────────────────────────────────────────
    print_comparison_table(histories, flops_dict=flops_dict)

    # Curve 1: Label smoothing effect on ResNet (B.2a vs B.2b)
    plot_comparison_curves(
        [h_b2a, h_b2b],
        title="ResNet — With vs Without Label Smoothing",
        save_path="b2_label_smoothing_curves.png",
    )

    # Curve 2: KD effect on SimpleCNN (B.1 vs B.3)
    plot_comparison_curves(
        [h_b1, h_b3],
        title="SimpleCNN — Baseline vs Knowledge Distillation",
        save_path="b3_kd_curves.png",
    )

    # Curve 3: Full comparison (all 5 runs)
    plot_comparison_curves(
        histories,
        title="Part B — All Experiments",
        save_path="b_all_curves.png",
    )


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