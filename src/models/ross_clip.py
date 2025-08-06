import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf.base import ContainerMetadata
from typing import List, Any
import os

from .clip import RegionCLIP
from .diffusion import ModifiedStableDiffusionInpaintPipeline
from ..training.scheduler import CosineAnnealingWarmupLR
from .utils import create_mask_from_bboxes, convert_bboxes_list_to_tensor
from ..data.dataloader import get_dataloader_len


class RossCLIP(nn.Module):
    def __init__(self, cfg):
        super(RossCLIP, self).__init__()
        self.clip = RegionCLIP(
            model_name=cfg["clip"]["pretrained"]["model_name"],
            pretrained=cfg["clip"]["pretrained"]["org"],
            iou_threshold=cfg["data"]["iou_threshold"],
            max_condition_length=cfg["data"]["max_condition_length"],
            use_pretrained=cfg["clip"]["use_pretrained"],
            load_weights=cfg["clip"]["pretrained"]["load_weights"],
            device="cuda",  # Pytorch Lightning will handle device placement
            mode = "train",
            embed_dim=cfg["clip"]["embed_dim"],
            image_resolution=cfg["clip"]["image_resolution"],
            vision_layers=cfg["clip"]["vision_layers"],
            vision_width=cfg["clip"]["vision_width"],
            vision_patch_size=cfg["clip"]["vision_patch_size"],
            context_length=cfg["clip"]["context_length"],
            vocab_size=cfg["clip"]["vocab_size"],
            transformer_width=cfg["clip"]["transformer_width"],
            transformer_heads=cfg["clip"]["transformer_heads"],
            transformer_layers=cfg["clip"]["transformer_layers"],
        )

        self.ross = ModifiedStableDiffusionInpaintPipeline(
            cfg["diffusion"]["pretrained_model_name_or_path"]
        )

        self.condition_projector = nn.Linear(
            in_features=self.clip.clip.visual.conv1.out_channels, out_features=self.ross.pipeline.text_encoder.config.hidden_size
        )

        self.gamma = cfg["loss"]["gamma"]  # Hyperparameter for loss weighting

        self.cfg = cfg

    def forward(self, images, captions, bboxes):
        """
        images: Tensor of shape (B, 3, H, W)
        captions: List of strings, one for each image in the batch
        bboxes: Tensor of shape (B, N, 4) where N is the number of bounding boxes or list of bounding boxes
        """
        if isinstance(bboxes, list):
            bboxes = convert_bboxes_list_to_tensor(bboxes)
        selected_features, global_features = self.clip.encode_image(images, bboxes)
        text_features = self.clip.encode_text(captions)
        diffusion_loss = self.ross(
            images,
            create_mask_from_bboxes(bboxes, images.shape[-1]),
            self.condition_projector(selected_features),
        )
        contrastive_loss = self.clip.contrastive_loss(global_features, text_features)
        return {
            "diffusion_loss": diffusion_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": diffusion_loss + self.gamma * contrastive_loss,
        }

    @torch.no_grad()
    def generate(self, images, captions, bboxes):
        """
        Generate images based on the input images, captions, and bounding boxes.
        """
        if isinstance(bboxes, list):
            bboxes = convert_bboxes_list_to_tensor(bboxes)
        selected_features, _ = self.clip.encode_image(images, bboxes)
        condition = self.condition_projector(selected_features)
        return self.ross.generate(
            images,
            create_mask_from_bboxes(bboxes, images.shape[-1]),
            condition,
            self.cfg["diffusion"]["num_inference_steps"],
            self.cfg["diffusion"]["strength"],
        )


def build_model(cfg):
    return RossCLIP(cfg)


class RossCLIPWrapper(LightningModule):
    def __init__(self, cfg):
        super(RossCLIPWrapper, self).__init__()
        self.model = build_model(cfg)
        self.cfg = cfg

    def forward(self, images, captions, bboxes):
        return self.model(images, captions, bboxes)

    def on_train_epoch_start(self):
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        images, captions, bboxes = batch["images"], batch["captions"], batch["bboxes"]
        output = self(images, captions, bboxes)

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, sync_dist=True)

        self.log(
            "train_loss",
            output["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "diffusion_loss",
            output["diffusion_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "contrastive_loss",
            output["contrastive_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return output["total_loss"]

    def configure_optimizers(self):
        optim_params = _get_optim_params(self.cfg, self.model)

        optimizer = torch.optim.AdamW(
            params=optim_params,
            lr=float(self.cfg["train"]["lr"]),
            weight_decay=float(self.cfg["train"]["weight_decay"]),
            betas=tuple(self.cfg["train"]["betas"]),
            eps=float(self.cfg["train"]["eps"]),
        )

        scheduler_config = self.cfg["train"].get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine_warmup")

        if scheduler_type == "cosine_warmup":
            epochs = self.cfg["train"]["epochs"]

            scheduler = CosineAnnealingWarmupLR(
                optimizer=optimizer,
                epochs=epochs,
                niter_per_ep=get_dataloader_len(self.cfg)
                // scheduler_config.get("frequency", 1),
                base_lr=float(scheduler_config.get("base_lr", self.cfg["train"]["lr"])),
                final_lr=float(scheduler_config.get("final_lr", 1e-6)),
                warmup_epochs=int(scheduler_config.get("warmup_epochs", 0)),
                start_warmup_lr=float(scheduler_config.get("start_warmup_lr", 0)),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": scheduler_config.get("frequency", 1),
                    "name": "cosine_warmup_lr",
                },
            }

        elif scheduler_type == "cosine_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(scheduler_config.get("T_0", 10)),
                T_mult=int(scheduler_config.get("T_mult", 2)),
                eta_min=float(scheduler_config.get("eta_min", 1e-6)),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": scheduler_config.get("frequency", 1),
                    "name": "cosine_restart_lr",
                },
            }

        else:
            return optimizer


def load_pretrained_model(
    cpkt_path, cfg, weights_only=False, device="cuda", mode="eval"
):
    """
    Load a pretrained model from a checkpoint file.

    Args:
        cpkt_path: Path to checkpoint file
        cfg: Model configuration
        weights_only: If True, only load weights safely. If None, auto-detect based on file
    """
    model = build_model(cfg).to(device)

    if cpkt_path is None or not os.path.exists(cpkt_path):
        return model

    try:
        # try to load the checkpoint with safe globals
        if weights_only is True:
            # use safe loading
            with torch.serialization.safe_globals([ContainerMetadata]):
                state_dict = torch.load(
                    cpkt_path, map_location="cpu", weights_only=True
                )
        elif weights_only is False:
            # use unsafe loading
            state_dict = torch.load(cpkt_path, map_location="cpu", weights_only=False)
        else:
            # auto-detect based on file content
            try:
                with torch.serialization.safe_globals([ContainerMetadata]):
                    state_dict = torch.load(
                        cpkt_path, map_location="cpu", weights_only=True
                    )
                print("Successfully loaded checkpoint with weights_only=True")
            except Exception as safe_error:
                print(f"Safe loading failed: {safe_error}")
                print(
                    "Falling back to weights_only=False (ensure you trust the checkpoint source)"
                )
                state_dict = torch.load(
                    cpkt_path, map_location="cpu", weights_only=False
                )

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {cpkt_path}: {e}")

    if "state_dict" in state_dict:
        model_state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        model_state_dict = state_dict["model_state_dict"]
    else:
        model_state_dict = state_dict

    if any(key.startswith("model.") for key in model_state_dict.keys()):
        cleaned_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith("model."):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value
        model_state_dict = cleaned_state_dict

    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state_dict, strict=False
        )

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

        print(f"Successfully loaded model from {cpkt_path}")

        if mode == "eval":
            model.eval()
        elif mode == "train":
            model.train()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'eval' or 'train'.")
        model.to(device)
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {e}")


def _get_optim_params(args, model) -> List[dict[str, Any]]:
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    return [
        {"params": p_wd, "weight_decay": args["train"]["weight_decay"]},
        {"params": p_non_wd, "weight_decay": 0.0},
    ]