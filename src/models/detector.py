from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    ZeroShotObjectDetectionPipeline,
)
from transformers.modeling_outputs import BaseModelOutput

class RelativePixelZeroShotObjectDetectionPipeline(ZeroShotObjectDetectionPipeline):
    def postprocess(self, model_outputs, threshold=0.1, top_k=None):
        results = []
        for model_output in model_outputs:
            label = model_output["candidate_label"]
            model_output = BaseModelOutput(model_output)
            outputs = self.image_processor.post_process_object_detection(
                outputs=model_output, threshold=threshold, target_sizes=None
            )[0]

            for index in outputs["scores"].nonzero():
                score = outputs["scores"][index].item()
                box = self._get_bounding_box(outputs["boxes"][index][0])

                result = {"score": score, "label": label, "box": box}
                results.append(result)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results
    
    def _get_bounding_box(self, box: torch.Tensor) -> dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`dict[str, float]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.float().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox

def init_zero_shot_object_detection_pipeline(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    detector = RelativePixelZeroShotObjectDetectionPipeline(model=model, processor=processor, device=device)
    return detector

# Copied from transformers.models.owlvit.image_processing_owlvit._scale_boxes
def scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes

@dataclass
class BoundingBox:
    xmin: Union[float, int]
    xmax: Union[float, int]
    ymin: Union[float, int]
    ymax: Union[float, int]

    @property
    def width(self) -> Union[float, int]:
        return self.xmax - self.xmin

    @property
    def height(self) -> Union[float, int]:
        return self.ymax - self.ymin

    @property
    def location(self) -> List[Union[float, int]]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    def to_list(self) -> List[Union[float, int]]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    def scale(self, target_size) -> "BoundingBox":
        """
        Scale the bounding box to the target size.

        Args:
            target_size (tuple): Target size as (height, width).

        Returns:
            BoundingBox: Scaled bounding box.
        """
        if isinstance(target_size, (list, tuple)):
            height, width = target_size
        else:
            raise TypeError("`target_size` must be a list or tuple")
        
        if not self.is_relative():
            self.xmin = int(self.xmin)
            self.xmax = int(self.xmax)
            self.ymin = int(self.ymin)
            self.ymax = int(self.ymax)
            return self
        
        self.xmin = int(self.xmin * width)
        self.xmax = int(self.xmax * width)
        self.ymin = int(self.ymin * height)
        self.ymax = int(self.ymax * height)

        return self
    
    def _between_zero_one(self, value):
        return value >= 0 and value <= 1
    
    def is_relative(self) -> bool:
        """
        Check if the bounding box coordinates are in relative format (between 0 and 1).

        Returns:
            bool: True if the bounding box is in relative format, False otherwise.
        """
        return self._between_zero_one(self.xmin) and self._between_zero_one(self.xmax) and \
               self._between_zero_one(self.ymin) and self._between_zero_one(self.ymax)


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        if "box" in detection_dict:
            box = detection_dict["box"]
            bbox = BoundingBox(
                xmin=float(box["xmin"]),
                xmax=float(box["xmax"]),
                ymin=float(box["ymin"]),
                ymax=float(box["ymax"])
            )
        elif "bbox" in detection_dict:
            bbox = BoundingBox(
                xmin=float(detection_dict["bbox"][0]),
                xmax=float(detection_dict["bbox"][2]),
                ymin=float(detection_dict["bbox"][1]),
                ymax=float(detection_dict["bbox"][3])
            )
        else:
            raise ValueError("Detection dictionary must contain 'box' or 'bbox' key.")
        return cls(
            score=float(detection_dict["score"]),
            label=detection_dict["label"],
            box=bbox,
            mask=detection_dict.get("mask", None)
        )

class Detector:
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
    ):
        self.detector_id = detector_id
        self.device = device
        self.detector = init_zero_shot_object_detection_pipeline(detector_id, device)

    def __call__(
        self, image: Image.Image, labels: List[str], threshold: float = 0.3
    ) -> List[DetectionResult]:
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.detector(
            image,
            candidate_labels=labels,
            threshold=threshold,
        )
        results = [DetectionResult.from_dict(result) for result in results]
        return results

def detect(image, labels: List[str], detector_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda", threshold: float = 0.5) -> List[DetectionResult]:
    detector = Detector(detector_id=detector_id, device=device)
    return detector(image, labels, threshold)

        
if __name__ == "__main__":
    from src.utils.plot_utils import plot_detections
    from src.models.noun_extractor import NounExtractor
    from PIL import Image
    import os

    example_image_path = "../data/000000004.jpg"
    example_text_path = "../data/000000004.txt"

    if not os.path.exists(example_image_path):
        raise FileNotFoundError(f"Image file {example_image_path} does not exist.")

    if not os.path.exists(example_text_path):
        raise FileNotFoundError(f"Text file {example_text_path} does not exist.")

    example_image = Image.open(example_image_path)

    with open(example_text_path, "r") as f:
        example_text = f.read().strip()
    
    noun_extractor = NounExtractor()
    nouns = noun_extractor.extract_nouns(example_text)
    print("Extracted Nouns:", nouns)
    device="cuda:1"
    detector = Detector(device=device)
    results = detector(example_image, nouns, threshold=0.3)
    print("Detection Results:")
    for result in results:
        print(f"Label: {result.label}, Score: {result.score}, Box: {result.box.location}")
    plot_detections(example_image, results, save_name="example_detections.png")