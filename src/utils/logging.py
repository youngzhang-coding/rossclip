import os
from omegaconf import DictConfig
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger


def is_main_process() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    return local_rank == 0 and rank == 0


def setup_logger(cfg: DictConfig) -> None | SwanLabLogger:
    try:
        print("Logging in to SwanLab...")
        swanlab.login(api_key=cfg["logger"]["api_key"])
    except Exception as e:
        print(f"Failed to login to SwanLab: {e}")
        return
    logger = None
    if is_main_process():
        try:
            logger = SwanLabLogger(
                project=cfg["logger"]["project_name"],
                workspace=cfg["logger"]["workspace"],
                experiment_name=cfg["logger"]["experiment_name"],
                logdir=cfg["logger"]["logdir"],
                config=cfg,
            )
        except Exception as e:
            print(f"Failed to initialize logger: {e}")
            logger = None
    return logger
