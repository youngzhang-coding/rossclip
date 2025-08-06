import argparse
import os
from PIL import Image
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.detector import detect
from src.models.noun_extractor import extract_nouns_from_text
from src.utils.plot_utils import plot_detections


def detect_entities(
    input_path,
    output_path,
    detector_id,
    spacy_language_model_id,
    device,
    threshold,
    verbose,
):
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image file {input_path} does not exist.")

    img = Image.open(input_path)

    img_name = os.path.basename(input_path)
    print(f"Processing image: {img_name}")

    text_path = input_path.replace(".jpg", ".txt")

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file {text_path} does not exist.")

    with open(text_path, "r") as f:
        text = f.read().strip()

    nouns = extract_nouns_from_text(text, model=spacy_language_model_id)

    print("Extracted Nouns:", nouns)

    results = detect(
        img, nouns, detector_id=detector_id, device=device, threshold=threshold
    )

    if verbose:
        print("Detection Results:")
        for result in results:
            print(f"Label: {result.label}, Score: {result.score}, Box: {result.box}")

    detection_filename = f"{img_name.split('.')[0]}_detections.jpg"
    plot_detections(
        img,
        results,
        save_name=os.path.join(output_path, detection_filename),
        display=True,
    )

    print(f"Detection results saved to {os.path.join(output_path, detection_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect entities in an image.")

    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="detected_results",
        help="Path to save the output results.",
    )
    parser.add_argument(
        "--detector_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="ID of the grounding detector to use.",
    )
    parser.add_argument(
        "--spacy_language_model_id",
        type=str,
        default="en_core_web_sm",
        help="ID of the spaCy language model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the detection on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold for the detector.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    detect_entities(
        input_path=args.input_path,
        output_path=args.output_path,
        detector_id=args.detector_id,
        spacy_language_model_id=args.spacy_language_model_id,
        device=args.device,
        threshold=args.threshold,
        verbose=args.verbose,
    )
