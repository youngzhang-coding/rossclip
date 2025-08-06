from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)
from torchvision.transforms.functional import InterpolationMode
import yaml

from src.constant import CONFIG_PATH


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def get_transform():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        n_px = config["clip"]["image_resolution"]
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
