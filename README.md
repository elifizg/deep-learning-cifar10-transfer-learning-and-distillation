# Transfer Learning and Knowledge Distillation on CIFAR-10

CS515 Deep Learning — Homework 1b

---

## Overview

This repository implements two sets of experiments on the CIFAR-10 dataset:

- **Part A — Transfer Learning:** Adapts an ImageNet-pretrained ResNet-18 to CIFAR-10 using two strategies (freeze backbone vs full fine-tuning).
- **Part B — Knowledge Distillation:** Trains compact student networks (SimpleCNN, MobileNetV2) using knowledge distillation and label smoothing, with ResNet-18 as the teacher.

---

## Results

| Experiment | Test Acc | FLOPs |
|---|---|---|
| Transfer — Option 1 (Freeze) | 79.11% | — |
| Transfer — Option 2 (Fine-tune) | 92.64% | — |
| SimpleCNN baseline | 76.77% | 6.28 MMac |
| ResNet-18 (no label smoothing) | 91.49% | 557.22 MMac |
| ResNet-18 (label smoothing ε=0.1) | 92.17% | 557.22 MMac |
| SimpleCNN + KD (T=4, α=0.3) | 76.80% | 6.28 MMac |
| MobileNetV2 + hybrid KD+LS | 89.05% | 96.16 MMac |

---

## Project Structure

```
.
├── main.py           # Single entry point for all experiments
├── train.py          # Training loop, data loading, loss functions
├── test.py           # Evaluation, per-class accuracy, FLOPs counting
├── pretrained.py     # Transfer learning (Option 1 and Option 2)
├── parameters.py     # Argparse + TrainingConfig dataclass
├── plot_results.py   # Generates all comparison plots for Part B
├── ensemble.py       # Ensemble utilities
├── models/
│   ├── CNN.py        # SimpleCNN and MNIST_CNN
│   ├── ResNet.py     # ResNet-18 (BasicBlock)
│   ├── VGG.py        # VGG-11/13/16/19
│   ├── MLP.py        # MLP for MNIST
│   └── mobilenet.py  # MobileNetV2
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/elifizg/deep-learning-cifar10-transfer-learning-and-distillation.git
cd deep-learning-cifar10-transfer-learning-and-distillation
pip install -r requirements.txt
pip install ptflops
```

---

## Usage

### Part A — Transfer Learning

Run both Option 1 and Option 2 and compare results:

```bash
python main.py --mode transfer --model resnet18 --epochs 20 \
               --transfer_option 0 --lr 1e-4 --batch_size 128 --tsne
```

| Argument | Description |
|---|---|
| `--model` | `resnet18` or `vgg16` |
| `--transfer_option` | `0` = both, `1` = freeze backbone, `2` = full fine-tune |
| `--tsne` | Generate t-SNE feature space plots |

---

### Part B — Knowledge Distillation

Run all five experiments in sequence (B1 → B2a → B2b → B3 → B4):

```bash
python main.py --mode kd --epochs 30
```

Or run individual experiments:

```bash
python main.py --mode b1    # SimpleCNN baseline
python main.py --mode b2a   # ResNet no label smoothing
python main.py --mode b2b   # ResNet with label smoothing
python main.py --mode b3    # SimpleCNN + KD (requires best_resnet.pth)
python main.py --mode b4    # MobileNet + hybrid KD (requires best_resnet.pth)
```

> **Note:** B3 and B4 require `best_resnet.pth` produced by B2a. Always run B2a before B3/B4.

---

### Generate Plots

After running all Part B experiments:

```bash
python plot_results.py
```

Produces:
- `b_final_comparison.png` — bar chart of test accuracy
- `b2_label_smoothing.png` — ResNet with vs without label smoothing
- `b3_kd_effect.png` — SimpleCNN baseline vs KD
- `b_all_curves.png` — all 5 experiments on one plot
- `b_flops_accuracy.png` — FLOPs vs accuracy scatter plot

---

## Saved Models

Best checkpoints are saved automatically during training:

| File | Description |
|---|---|
| `best_resnet18_option1.pth` | Transfer learning Option 1 |
| `best_resnet18_option2.pth` | Transfer learning Option 2 |
| `best_cnn_baseline.pth` | SimpleCNN baseline (B.1) |
| `best_resnet.pth` | ResNet-18 no LS (B.2a) — used as KD teacher |
| `best_resnet_ls.pth` | ResNet-18 with LS (B.2b) |
| `best_cnn_kd.pth` | SimpleCNN + KD student (B.3) |
| `best_mobilenet_kd.pth` | MobileNetV2 hybrid KD student (B.4) |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib
scikit-learn
ptflops
```

---

## References

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
2. Hinton et al., *Distilling the Knowledge in a Neural Network*, arXiv 2015
3. Müller et al., *When Does Label Smoothing Help?*, NeurIPS 2019
4. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018