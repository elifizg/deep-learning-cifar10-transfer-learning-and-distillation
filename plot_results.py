"""
plot_results.py
===============
Generates all comparison plots for Part B after experiments are complete.

Produces:
  1. b_final_comparison.png  — bar chart of final test accuracy (all 5 runs)
  2. b2_label_smoothing.png  — val accuracy curves: ResNet no-LS vs LS=0.1
  3. b3_kd_effect.png        — val accuracy curves: SimpleCNN baseline vs KD
  4. b_all_curves.png        — all 5 experiments on one plot
  5. b_flops_accuracy.png    — FLOPs vs accuracy scatter plot

Reads:
  - history_*.json  (saved automatically by run_training_tracked)
  - best_*.pth      (for FLOPs measurement)

Usage:
    python plot_results.py
"""

import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.CNN      import SimpleCNN
from models.ResNet   import ResNet, BasicBlock
from models.mobilenet import MobileNetV2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
COLORS       = ["#4f86c6", "#e07b39", "#f5c842", "#5aab61", "#c45c8a"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_history(json_path: str) -> dict:
    """Load a RunHistory dict from a JSON file saved by run_training_tracked."""
    with open(json_path) as f:
        return json.load(f)


def get_flops(model: nn.Module) -> str:
    """Return FLOPs string via ptflops, or param count fallback."""
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        return macs
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return f"{params / 1e6:.2f} M params"


def get_flops_numeric(model: nn.Module) -> float:
    """Return FLOPs as a float (MMac) for scatter plot axis."""
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, 32, 32), as_strings=False,
            print_per_layer_stat=False, verbose=False)
        return macs / 1e6   # MMac
    except Exception:
        return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return test accuracy."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar(labels: List[str], accs: List[float], save: str) -> None:
    """Bar chart of final test accuracy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, [a * 100 for a in accs],
                  color=COLORS[:len(labels)], width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc*100:.2f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Part B — Knowledge Distillation: Final Test Accuracy")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved: {save}")


def plot_curves(histories: List[dict], title: str, save: str) -> None:
    """Validation accuracy curves for multiple experiments."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for i, h in enumerate(histories):
        c      = COLORS[i % len(COLORS)]
        epochs = list(range(1, len(h["val_acc"]) + 1))
        axes[0].plot(epochs, h["train_loss"], marker="o", color=c, label=h["label"])
        axes[1].plot(epochs, h["val_acc"],    marker="o", color=c, label=h["label"])

    for ax, ylabel, ytitle in zip(
        axes,
        ["Loss", "Accuracy"],
        ["Training Loss per Epoch", "Validation Accuracy per Epoch"],
    ):
        ax.set_title(ytitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if epochs:
            ax.set_xticks(epochs)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True)
        ax.grid(linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


def plot_flops_scatter(labels: List[str], accs: List[float],
                       flops: List[float], save: str) -> None:
    """Scatter plot: FLOPs (x) vs Test Accuracy (y)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, acc, flop) in enumerate(zip(labels, accs, flops)):
        c = COLORS[i % len(COLORS)]
        ax.scatter(flop, acc * 100, s=150, color=c, zorder=3)
        ax.annotate(label, (flop, acc * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("FLOPs (MMac)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Part B — FLOPs vs Accuracy Trade-off")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved: {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")   # headless rendering for Colab

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # ── Load histories from JSON ──────────────────────────────────────────────
    # File names match the label sanitised by run_training_tracked()
    history_files = {
        "SimpleCNN (baseline)":     "history_SimpleCNN__baseline_.json",
        "ResNet (no LS)":           "history_ResNet__no_LS_.json",
        "ResNet (LS=0.1)":          "history_ResNet__LS_0_1_.json",
        "SimpleCNN (KD)":           "history_SimpleCNN__KD_.json",
        "MobileNet (hybrid KD+LS)": "history_MobileNet__hybrid_KD_LS_.json",
    }

    histories: Dict[str, dict] = {}
    for label, fname in history_files.items():
        if os.path.exists(fname):
            histories[label] = load_history(fname)
            print(f"  Loaded: {fname}")
        else:
            print(f"  Missing: {fname}  (skipped)")

    # ── Load models for FLOPs + test accuracy ─────────────────────────────────
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_ds     = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model_defs = [
        ("SimpleCNN (baseline)", SimpleCNN(num_classes=10),                      "best_cnn_baseline.pth"),
        ("ResNet (no LS)",       ResNet(BasicBlock, [2,2,2,2], num_classes=10),  "best_resnet.pth"),
        ("ResNet (LS=0.1)",      ResNet(BasicBlock, [2,2,2,2], num_classes=10),  "best_resnet_ls.pth"),
        ("SimpleCNN (KD)",       SimpleCNN(num_classes=10),                      "best_cnn_kd.pth"),
        ("MobileNet (hybrid)",   MobileNetV2(num_classes=10),                    "best_mobilenet_kd.pth"),
    ]

    labels, accs, flops_str, flops_num = [], [], [], []

    for label, model, path in model_defs:
        if not os.path.exists(path):
            print(f"  {label:<28}  SKIPPED (no weights: {path})")
            continue
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        acc  = evaluate(model, test_loader, device)
        fs   = get_flops(model)
        fn   = get_flops_numeric(model)
        labels.append(label)
        accs.append(acc)
        flops_str.append(fs)
        flops_num.append(fn)
        model.cpu()
        torch.cuda.empty_cache()
        print(f"  {label:<28}  acc={acc:.4f}  flops={fs}")

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"  {'Experiment':<28} {'Test Acc':>9}  {'FLOPs':>18}")
    print(f"{'─' * 68}")
    for label, acc, flops in zip(labels, accs, flops_str):
        print(f"  {label:<28} {acc:>8.4f}  {flops:>18}")
    print(f"{'=' * 68}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    # 1. Bar chart — all experiments
    if labels:
        plot_bar(labels, accs, "b_final_comparison.png")

    # 2. Label smoothing curves — B.2a vs B.2b
    ls_histories = [h for k, h in histories.items()
                    if k in ("ResNet (no LS)", "ResNet (LS=0.1)")]
    if len(ls_histories) == 2:
        plot_curves(ls_histories,
                    "ResNet — With vs Without Label Smoothing",
                    "b2_label_smoothing.png")

    # 3. KD effect curves — B.1 vs B.3
    kd_histories = [h for k, h in histories.items()
                    if k in ("SimpleCNN (baseline)", "SimpleCNN (KD)")]
    if len(kd_histories) == 2:
        plot_curves(kd_histories,
                    "SimpleCNN — Baseline vs Knowledge Distillation",
                    "b3_kd_effect.png")

    # 4. All 5 curves together
    all_h = [h for k, h in histories.items()]
    if all_h:
        plot_curves(all_h, "Part B — All Experiments", "b_all_curves.png")

    # 5. FLOPs vs accuracy scatter
    if flops_num:
        plot_flops_scatter(labels, accs, flops_num, "b_flops_accuracy.png")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()