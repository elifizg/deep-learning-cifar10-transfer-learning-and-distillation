import argparse


def get_params():
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    args = parser.parse_args()

    return {
        # Data
        "data_dir":     "./data",
        "num_workers":  2,

        # Model
        "input_size":   784,        # 28x28
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "best_model.pth",
        "log_interval": 100,        # print every N batches

        # CLI
        "mode":         args.mode,
    }