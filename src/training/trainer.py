from omegaconf import DictConfig
from lightning import Trainer as LightningTrainer

from src.models import RossCLIPWrapper
from src.data.dataloader import get_train_dataloader
from src.utils.logging import setup_logger


class RossCLIPTrainer(LightningTrainer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = setup_logger(cfg)
        super().__init__(
            max_epochs=cfg["train"]["epochs"],
            devices=cfg["train"]["gpu_ids"],
            logger=self.logger,
            accelerator="auto",
            precision=cfg["train"]["precision"],
        )

    def setup_model(self):
        return RossCLIPWrapper(self.cfg)

    def train(self):
        model = self.setup_model()
        dataloader = get_train_dataloader(self.cfg)
        self.fit(model, dataloader)
        return self, model