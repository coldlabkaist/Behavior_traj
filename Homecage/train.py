import os
from train.trainer import Trainer
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config', type=str, default='cfg/train_config.yaml', help='Path to the training configuration file.')
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        print(
            "Error: Multi-process execution is disabled for reproducibility. "
            "Run a single process (e.g., `python train.py`)."
        )
        return

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
