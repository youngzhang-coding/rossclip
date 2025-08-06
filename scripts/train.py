import hydra
from omegaconf import DictConfig
from lightning import seed_everything
import torch
import datetime

from src.training.trainer import RossCLIPTrainer
from src.constant import CONFIG_DIR

@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to start training.
    """
    seed_everything(cfg["train"]["seed"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Initializing RossCLIP Trainer at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on gpus {cfg['train']['gpu_ids']} with precision {cfg['train']['precision']}"
    )

    try:
        trainer = RossCLIPTrainer(cfg)
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")
        return
    print(
        f"Trainer initialized at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        return

    print(
        f"Training completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
