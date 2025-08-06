import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict
from src.models.detector import DetectionResult
from src.data.preprocess import get_transform
from src.models.tokenizer import tokenize
from omegaconf import DictConfig


class JsonDataset(Dataset):
    def __init__(self, path, transform=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset info file {path} not found.")
        with open(path, "r") as f:
            self.info = json.load(f)
        self.info = list(self.info.items())
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        item = self.info[idx]
        image_path = item[0]
        bbox_info = item[1]
        perfix_path = os.path.splitext(image_path)[0]
        caption_path = perfix_path + ".txt"
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file {caption_path} not found.")
        with open(caption_path, "r") as f:
            caption = f.read().strip()
        image = Image.open(image_path)
        return DataItem(
            image=image,
            caption=caption,
            bbox=[
                DetectionResult.from_dict(detection_dict)
                for detection_dict in bbox_info
            ],
        )

@dataclass
class DataItem:
    image: Image.Image
    caption: str
    bbox: List[DetectionResult]

    def get_bbox(self, idx: int) -> List[int]:
        if idx < 0 or idx >= len(self.bbox):
            raise IndexError(f"Index {idx} out of range for bounding boxes.")
        bbox = self.bbox[idx].box
        return [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

    def get_top_k_bbox(self, k: int) -> List[List[int]]:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        bbox = sorted(self.bbox, key=lambda x: x.score, reverse=True)
        res = []
        tmp = min(k, len(bbox))
        for i in range(tmp):
            res.append(bbox[i].box.to_list())

        if len(res) < k:
            res.extend([[0, 0, 0, 0]] * (k - len(res)))
        return res


def collate_fn(batch: List[DataItem], cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for JsonDataset that processes batches of DataItem objects.

    Args:
        batch: List of DataItem objects containing:
            - image: PIL.Image
            - caption: str
            - bbox: List[DetectionResult]

    Returns:
        Dictionary containing batched tensors and metadata:
        {
            "images": tensor (B, C, H, W),   # Batched image tensors
            "captions": List[torch.Tensor],  # tokenized captions for each sample
            "bboxes": tensor(B, K, 4),       # top k bbox coordinates per sample
        }
    """
    # Initialize containers
    images, captions, bboxes = [], [], []

    # Process each DataItem in batch
    for item in batch:
        # --- Image Processing ---
        # Convert to RGB if needed (e.g., grayscale -> RGB)
        img_tensor = get_transform()(item.image)
        images.append(img_tensor)

        # --- Text Processing ---
        caption_tokens = tokenize(
            item.caption,
            context_length=cfg["clip"]["context_length"],
            truncate=True,
        )
        captions.append(caption_tokens)

        # --- Bounding Box Processing ---
        # Extract coordinates and confidence scores
        current_bboxes = item.get_top_k_bbox(
            cfg["data"]["top_k"]
        )  # List Shape: (K, 4)
        bboxes.append(current_bboxes)

    # --- Batch Alignment ---
    # Stack images into (B, C, H, W) tensor (requires consistent dimensions)
    images = torch.stack(images)
    captions = torch.stack(captions).squeeze(1)  # (B, context_length)
    bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.int32)  # (B, K, 4)

    return {
        "images": images,  # (B, C, H, W)
        "captions": captions,  # (B, context_length)
        "bboxes": bboxes,  # (B, K, 4)
    }

def get_train_dataloader(cfg: DictConfig):
    dataset = JsonDataset(path=cfg["data"]["data_info_path"], transform=None)

    def collate_fn_wrapper(batch):
        return collate_fn(batch, cfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn_wrapper,
        pin_memory=True,
    )
    return dataloader

def get_dataloader_len(cfg: DictConfig) -> int:
    """
    Returns the length of the training dataloader.
    """
    dataloader = get_train_dataloader(cfg)
    return len(dataloader)