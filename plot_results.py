"""
plot_results.py
===============
Generates comparison plots for Part B after all experiments are complete.

Loads each saved model, evaluates on the test set, and produces:
  1. Bar chart   — final test accuracy comparison (all 5 experiments)
  2. FLOPs table — parameter and compute cost comparison

Usage (after all b1-b4 experiments have run):
    python plot_results.py --epochs 20

Requires: best_cnn_baseline.pth, best_resnet.pth, best_resnet_ls.pth,
          best_cnn_kd.pth, best_mobilenet_kd.pth
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.CNN      import SimpleCNN
from models.ResNet   import ResNet, BasicBlock
from models.mobilenet import MobileNetV2


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return test accuracy for the given model and loader."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return correct / total


def get_flops(model: nn.Module) -> str:
    """Return FLOPs string using ptflops, or param count if unavailable."""
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        return macs
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return f"{params:,} params"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # Test loader
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_ds     = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Experiment definitions: (label, model_class, weights_path)
    experiments = [
        ("SimpleCNN (baseline)", SimpleCNN(num_classes=10),         "best_cnn_baseline.pth"),
        ("ResNet (no LS)",       ResNet(BasicBlock,[2,2,2,2],num_classes=10), "best_resnet.pth"),
        ("ResNet (LS=0.1)",      ResNet(BasicBlock,[2,2,2,2],num_classes=10), "best_resnet_ls.pth"),
        ("SimpleCNN (KD)",       SimpleCNN(num_classes=10),         "best_cnn_kd.pth"),
        ("MobileNet (hybrid)",   MobileNetV2(num_classes=10),       "best_mobilenet_kd.pth"),
    ]

    labels, accs, flops_list = [], [], []

    for label, model, path in experiments:
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            acc   = evaluate(model, test_loader, device)
            flops = get_flops(model)
            labels.append(label)
            accs.append(acc)
            flops_list.append(flops)
            model.cpu()
            torch.cuda.empty_cache()
            print(f"  {label:<28}  acc={acc:.4f}  flops={flops}")
        except FileNotFoundError:
            print(f"  {label:<28}  SKIPPED (weights not found: {path})")

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'Experiment':<28} {'Test Acc':>9}  {'FLOPs':>15}")
    print(f"{'─' * 65}")
    for label, acc, flops in zip(labels, accs, flops_list):
        print(f"  {label:<28} {acc:>8.4f}  {flops:>15}")
    print(f"{'=' * 65}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        colors = ["#4f86c6", "#e07b39", "#f5c842", "#5aab61", "#c45c8a"]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, [a * 100 for a in accs],
                      color=colors[:len(labels)], width=0.5)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc*100:.2f}%",
                    ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Part B — Knowledge Distillation: Final Test Accuracy")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig("b_final_comparison.png", dpi=150)
        plt.close()
        print("Bar chart saved to: b_final_comparison.png")

    except ImportError:
        print("matplotlib not found — skipping plot.")


if __name__ == "__main__":
    main()