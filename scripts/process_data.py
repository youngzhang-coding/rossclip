import os
import json
import logging
from typing import Tuple, Optional, Dict, List, Any
from tqdm import tqdm
from PIL import Image
import argparse
import torch

from src.models.detector import Detector
from src.models.noun_extractor import NounExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment(device: str) -> None:
    logger.info(f"Starting preprocessing on {device}")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return device


def initialize_models(
    detector_id: str, spacy_language_model_id: str, device: str
) -> Tuple[Detector, NounExtractor]:
    detector = Detector(detector_id, device)
    noun_extractor = NounExtractor(spacy_language_model_id)
    return detector, noun_extractor


def find_image_files(input_path: str) -> List[Tuple[str, str]]:
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = []

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append((root, file))

    logger.info(f"Found {len(image_files)} image files to process")
    return image_files


def process_single_image(
    args: Tuple[Detector, NounExtractor, str, str],
) -> Tuple[Optional[str], Any]:
    detector, noun_extractor, root, file = args
    abs_path = os.path.join(root, file)
    prefix = os.path.splitext(file)[0]
    text_file_path = os.path.join(root, prefix + ".txt")

    if not os.path.exists(text_file_path):
        raise FileNotFoundError(
            f"Image file {abs_path} has no corresponding text file {text_file_path}"
        )

    logger.info(f"Processing Image: {abs_path}")
    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return None, f"Empty text file {text_file_path}, skipping {abs_path}"

        nouns = noun_extractor.extract_nouns(text)
        if not nouns:
            return (
                None,
                f"No nouns extracted from {text_file_path}, skipping {abs_path}",
            )

        with Image.open(abs_path) as image:
            results = detector(image, nouns)

        result_items = []
        for result in results:
            result_item = {
                "label": result.label,
                "score": float(result.score),
                "bbox": [float(x) for x in result.box.location],
            }
            result_items.append(result_item)

        return abs_path, result_items

    except Exception as e:
        logger.error(f"Error processing {abs_path}: {str(e)}", exc_info=True)
        return None, f"Error processing {abs_path}: {str(e)}"


def process_images_serial(
    image_files: List[Tuple[str, str]],
    detector: Detector,
    noun_extractor: NounExtractor,
    output_path: str = None,
    save_interval: int = 10,
) -> Tuple[List[str], int, int]:
    processed_count = 0
    error_count = 0
    output_data = {}
    temp_files = []

    for idx, (root, file) in enumerate(tqdm(image_files, desc="Processing images")):
        result = process_single_image((detector, noun_extractor, root, file))
        if result[0] is not None:
            abs_path, result_items = result
            output_data[abs_path] = result_items
            processed_count += 1
        else:
            error_count += 1

        if output_path and (idx + 1) % save_interval == 0:
            temp_path = output_path[:-5] + f"_part_{idx + 1}.json"
            save_results(output_data, temp_path)
            temp_files.append(temp_path)
            logger.info(f"Temporary results saved to {temp_path}")
            output_data.clear()

    if output_data and output_path:
        temp_path = output_path + f"_part_final.json"
        save_results(output_data, temp_path)
        temp_files.append(temp_path)
        logger.info(f"Final temporary results saved to {temp_path}")

    return temp_files, processed_count, error_count


def save_results(output_data: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {output_path}")


def merge_temp_json_files(temp_files: List[str], output_path: str):
    merged_data = {}
    for temp_file in temp_files:
        with open(temp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.update(data)
    save_results(merged_data, output_path)
    logger.info(f"Merged {len(temp_files)} temp files into {output_path}")


def preprocess(
    input_path: str,
    output_path: str,
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    spacy_language_model_id: str = "en_core_web_sm",
    device: str = "cuda",
    save_interval: int = 100,
) -> None:
    device = setup_environment(device)

    detector, noun_extractor = initialize_models(
        detector_id, spacy_language_model_id, device
    )

    image_files = find_image_files(input_path)
    if not image_files:
        logger.warning("No image files found in the input path")
        return

    temp_files, processed_count, error_count = process_images_serial(
        image_files, detector, noun_extractor, output_path, save_interval
    )

    logger.info(
        f"Preprocessing completed. Processed: {processed_count}, Errors: {error_count}"
    )

    merge_temp_json_files(temp_files, output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess images and captions for object detection."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input directory containing images and captions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--detector_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face model ID for the detector.",
    )
    parser.add_argument(
        "--spacy_language_model_id",
        type=str,
        default="en_core_web_sm",
        help="Spacy language model ID for noun extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the detector on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    logger.setLevel(args.log_level)

    preprocess(
        args.input_path,
        args.output_path,
        args.detector_id,
        args.spacy_language_model_id,
        args.device,
    )
