import torch
import numpy as np
from src.models.ross_clip import load_pretrained_model
from omegaconf import DictConfig
from scripts.process_data import process_single_image
from src.models.detector import Detector
from src.models.noun_extractor import NounExtractor
from src.data.preprocess import get_transform
from src.models.tokenizer import tokenize
import os
from PIL import Image
from torchvision.transforms import ToPILImage
from typing import List, Dict, Any
import logging


def eval_image_reconstruction(
    input_path: str,
    output_path: str,
    cpkt_path: str,
    cfg: DictConfig,
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    spacy_language_model_id: str = "en_core_web_sm",
    weights_only: bool = True,
    device: str = "cuda",
) -> None:
    """Evaluate image reconstruction with robust dtype handling."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the pretrained model
    logger.info(f"Loading model from checkpoint: {cpkt_path}")
    model = load_pretrained_model(
        cpkt_path, cfg, weights_only, device=device, mode="eval"
    )
    model.eval()

    # Get actual model dtype from diffusion pipeline
    actual_model_dtype = get_actual_model_dtype(model)
    logger.info(f"Detected model dtype: {actual_model_dtype}")

    # Use the actual model dtype instead of assuming based on device
    expected_dtype = actual_model_dtype
    logger.info(f"Using dtype: {expected_dtype} for device: {device}")

    # Initialize detector and noun extractor
    logger.info(f"Initializing detector: {detector_id}")
    detector = Detector(detector_id=detector_id, device=device)

    logger.info(f"Initializing noun extractor: {spacy_language_model_id}")
    noun_extractor = NounExtractor(model=spacy_language_model_id)

    # Process the input image to get detections
    root, file = os.path.split(input_path)
    input_batch = (detector, noun_extractor, root, file)

    logger.info(f"Processing image: {input_path}")
    abs_path, result_items = process_single_image(input_batch)

    # Load and verify the image
    try:
        image = Image.open(abs_path).convert("RGB")
        logger.info(f"Successfully loaded image: {abs_path}")
        logger.info(f"Image size: {image.size}")
    except Exception as e:
        logger.error(f"Error opening image {abs_path}: {str(e)}")
        return

    # Check if we have detection results
    if not result_items:
        logger.warning(f"No detection results for {abs_path}")
        logger.info("Using default full-image bbox for reconstruction")
        result_items = [
            {"bbox": [0, 0, image.size[0], image.size[1]], "label": "image"}
        ]

    logger.info(f"Found {len(result_items)} detection results")

    # Process the image using the same transform as training
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Convert to expected dtype and move to device
    image_tensor = image_tensor.to(device=device, dtype=expected_dtype)
    logger.info(
        f"Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}"
    )

    # Extract bboxes and captions from detection results
    raw_bboxes = [item["bbox"] for item in result_items]
    raw_captions = [item["label"] for item in result_items]

    logger.info(f"Raw bboxes (first 5): {raw_bboxes[:5]}")
    logger.info(f"Raw captions (first 5): {raw_captions[:5]}")

    # Process bboxes to match training format
    processed_bboxes = process_bboxes_for_inference(
        raw_bboxes, image.size, top_k=cfg["data"]["top_k"]
    )

    # Process captions to match training format
    processed_captions = process_captions_for_inference(
        raw_captions, cfg["CLIP"]["context_length"]
    )

    logger.info(f"Processed bboxes shape: {processed_bboxes.shape}")
    logger.info(f"Processed captions shape: {processed_captions.shape}")

    # Prepare batch data with consistent dtypes
    batch_data = prepare_inference_batch(
        image_tensor, processed_captions, processed_bboxes, device, expected_dtype
    )

    # Debug data shapes and types
    debug_data_shapes(
        batch_data["images"], batch_data["captions"], batch_data["bboxes"]
    )

    # Ensure model is in correct dtype before generation
    ensure_model_dtype_consistency(model, expected_dtype, logger)

    # Generate reconstructed image with fallback mechanisms
    logger.info("Generating reconstructed image...")
    with torch.no_grad():
        try:
            reconstructed_images = model.generate(
                images=batch_data["images"],
                captions=batch_data["captions"],
                bboxes=batch_data["bboxes"],
            )
            logger.info(
                f"Reconstruction successful. Output shape: {reconstructed_images.shape}"
            )

        except RuntimeError as e:
            if "Input type" in str(e) and "bias type" in str(e):
                logger.warning(
                    "Data type mismatch detected. Attempting fallback to float32..."
                )

                # Fallback 1: Try with float32
                try:
                    batch_data_float32 = convert_batch_to_dtype(
                        batch_data, torch.float32
                    )
                    ensure_model_dtype_consistency(model, torch.float32, logger)

                    reconstructed_images = model.generate(
                        images=batch_data_float32["images"],
                        captions=batch_data_float32["captions"],
                        bboxes=batch_data_float32["bboxes"],
                    )
                    logger.info("Fallback to float32 successful")

                except RuntimeError as e2:
                    logger.warning("Float32 fallback failed. Trying mixed precision...")

                    # Fallback 2: Mixed precision approach
                    batch_data_mixed = convert_batch_mixed_precision(batch_data, device)
                    ensure_model_mixed_precision(model, logger)

                    reconstructed_images = model.generate(
                        images=batch_data_mixed["images"],
                        captions=batch_data_mixed["captions"],
                        bboxes=batch_data_mixed["bboxes"],
                    )
                    logger.info("Mixed precision fallback successful")
            else:
                logger.error(f"Error during image generation: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during image generation: {str(e)}")
            raise

    # Convert to PIL image and save
    save_reconstructed_image(reconstructed_images, output_path)
    logger.info(f"Reconstructed image saved to: {output_path}")


def get_actual_model_dtype(model: Any) -> torch.dtype:
    """
    Get the actual data type used by the model components.

    Args:
        model: The RossCLIP model

    Returns:
        The detected data type
    """
    try:
        # Check diffusion pipeline components
        if hasattr(model, "ross") and hasattr(model.ross, "pipeline"):
            pipeline = model.ross.pipeline

            # Check VAE dtype
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "parameters"):
                vae_param = next(pipeline.vae.parameters())
                return vae_param.dtype

            # Check UNet dtype
            if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "parameters"):
                unet_param = next(pipeline.unet.parameters())
                return unet_param.dtype

    except Exception as e:
        print(f"Warning: Could not detect model dtype: {e}")

    # Default fallback
    return torch.float32


def ensure_model_dtype_consistency(
    model: Any, target_dtype: torch.dtype, logger: logging.Logger
) -> None:
    """
    Ensure all model components use the same data type.

    Args:
        model: The RossCLIP model
        target_dtype: Target data type
        logger: Logger instance
    """
    try:
        if hasattr(model, "ross") and hasattr(model.ross, "pipeline"):
            pipeline = model.ross.pipeline

            # Convert VAE
            if hasattr(pipeline, "vae"):
                current_dtype = next(pipeline.vae.parameters()).dtype
                if current_dtype != target_dtype:
                    logger.info(
                        f"Converting VAE from {current_dtype} to {target_dtype}"
                    )
                    pipeline.vae = pipeline.vae.to(dtype=target_dtype)

            # Convert UNet
            if hasattr(pipeline, "unet"):
                current_dtype = next(pipeline.unet.parameters()).dtype
                if current_dtype != target_dtype:
                    logger.info(
                        f"Converting UNet from {current_dtype} to {target_dtype}"
                    )
                    pipeline.unet = pipeline.unet.to(dtype=target_dtype)

            # Convert text encoder if present
            if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
                current_dtype = next(pipeline.text_encoder.parameters()).dtype
                if current_dtype != target_dtype:
                    logger.info(
                        f"Converting text encoder from {current_dtype} to {target_dtype}"
                    )
                    pipeline.text_encoder = pipeline.text_encoder.to(dtype=target_dtype)

    except Exception as e:
        logger.warning(f"Could not ensure model dtype consistency: {e}")


def ensure_model_mixed_precision(model: Any, logger: logging.Logger) -> None:
    """
    Set up mixed precision for the model.

    Args:
        model: The RossCLIP model
        logger: Logger instance
    """
    try:
        if hasattr(model, "ross") and hasattr(model.ross, "pipeline"):
            pipeline = model.ross.pipeline

            # Keep VAE in float32 (often more stable)
            if hasattr(pipeline, "vae"):
                pipeline.vae = pipeline.vae.to(dtype=torch.float32)
                logger.info("Set VAE to float32")

            # UNet can stay in float16
            if hasattr(pipeline, "unet"):
                pipeline.unet = pipeline.unet.to(dtype=torch.float16)
                logger.info("Set UNet to float16")

            # Text encoder in float32
            if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
                pipeline.text_encoder = pipeline.text_encoder.to(dtype=torch.float32)
                logger.info("Set text encoder to float32")

    except Exception as e:
        logger.warning(f"Could not set up mixed precision: {e}")


def convert_batch_to_dtype(
    batch_data: Dict[str, torch.Tensor], target_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """
    Convert batch data to target dtype (except for text tokens).

    Args:
        batch_data: Dictionary containing batch tensors
        target_dtype: Target data type

    Returns:
        Dictionary with converted tensors
    """
    converted_batch = {}
    for key, tensor in batch_data.items():
        if key == "captions":
            # Keep text tokens as integers/long
            converted_batch[key] = tensor
        else:
            # Convert images and bboxes to target dtype
            converted_batch[key] = tensor.to(dtype=target_dtype)
    return converted_batch


def convert_batch_mixed_precision(
    batch_data: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    """
    Convert batch data using mixed precision strategy.

    Args:
        batch_data: Dictionary containing batch tensors
        device: Target device

    Returns:
        Dictionary with mixed precision tensors
    """
    mixed_batch = {}
    for key, tensor in batch_data.items():
        if key == "captions":
            # Keep text tokens as they are
            mixed_batch[key] = tensor.to(device)
        elif key == "images":
            # Images in float32 for VAE stability
            mixed_batch[key] = tensor.to(device=device, dtype=torch.float32)
        else:
            # Other tensors (bboxes) in float32
            mixed_batch[key] = tensor.to(device=device, dtype=torch.float32)
    return mixed_batch


def process_bboxes_for_inference(
    raw_bboxes: List[List[int]], image_size: tuple, top_k: int
) -> torch.Tensor:
    """
    Process bboxes to match the training format.

    Args:
        raw_bboxes: List of bboxes in [x1, y1, x2, y2] format
        image_size: (width, height) of the original image
        top_k: Maximum number of bboxes to keep

    Returns:
        Tensor of shape (1, top_k, 4) with normalized coordinates
    """
    width, height = image_size
    processed_bboxes = []

    # Take top_k bboxes (assuming they're already sorted by confidence)
    selected_bboxes = raw_bboxes[:top_k]

    for bbox in selected_bboxes:
        # Normalize coordinates to [0, 1] and clamp
        x1, y1, x2, y2 = bbox
        normalized_bbox = [
            max(0.0, min(1.0, x1 / width)),  # x1 - clamp to [0, 1]
            max(0.0, min(1.0, y1 / height)),  # y1 - clamp to [0, 1]
            max(0.0, min(1.0, x2 / width)),  # x2 - clamp to [0, 1]
            max(0.0, min(1.0, y2 / height)),  # y2 - clamp to [0, 1]
        ]
        processed_bboxes.append(normalized_bbox)

    # Pad with zeros if we have fewer than top_k bboxes
    while len(processed_bboxes) < top_k:
        processed_bboxes.append([0, 0, 0, 0])

    # Convert to tensor and add batch dimension
    bboxes_tensor = torch.tensor(processed_bboxes, dtype=torch.float32).unsqueeze(0)
    return bboxes_tensor


def process_captions_for_inference(
    raw_captions: List[str], context_length: int
) -> torch.Tensor:
    """
    Process captions to match the training format.

    Args:
        raw_captions: List of caption strings
        context_length: Maximum context length for tokenization

    Returns:
        Tensor of tokenized captions
    """
    # Remove duplicates while preserving order, then join
    seen = set()
    unique_captions = []
    for caption in raw_captions:
        if caption not in seen:
            unique_captions.append(caption)
            seen.add(caption)

    # Join unique captions (limit to avoid very long text)
    combined_caption = " ".join(unique_captions[:10])

    # Tokenize using the same function as training
    caption_tokens = tokenize(
        combined_caption,
        context_length=context_length,
        truncate=True,
    )

    return caption_tokens


def prepare_inference_batch(
    image_tensor: torch.Tensor,
    caption_tokens: torch.Tensor,
    bbox_tensor: torch.Tensor,
    device: str,
    expected_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Prepare batch data with consistent dtypes."""
    return {
        "images": image_tensor.to(device=device, dtype=expected_dtype),
        "captions": caption_tokens.unsqueeze(0).to(
            device=device
        ),  # Keep as long/int for tokens
        "bboxes": bbox_tensor.to(device=device, dtype=expected_dtype),
    }


def save_reconstructed_image(
    reconstructed_images: torch.Tensor, output_path: str
) -> None:
    """
    Save the reconstructed image tensor as a PIL image.

    Args:
        reconstructed_images: Tensor of reconstructed images
        output_path: Path to save the image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Handle batch dimension
    if reconstructed_images.dim() == 4:
        reconstructed_image = reconstructed_images[0]  # Take first image from batch
    else:
        reconstructed_image = reconstructed_images

    # Move to CPU and convert to float32 for PIL processing
    reconstructed_image = reconstructed_image.cpu().float()

    # Handle different value ranges
    if reconstructed_image.min() < 0:
        # Convert from [-1, 1] to [0, 1]
        reconstructed_image = (reconstructed_image + 1) / 2

    # Clamp values to valid range
    reconstructed_image = torch.clamp(reconstructed_image, 0, 1)

    # Convert to PIL image
    to_pil = ToPILImage()
    pil_image = to_pil(reconstructed_image)

    # Save the image
    pil_image.save(output_path)
    print(f"Image saved with size: {pil_image.size}")


def debug_data_shapes(
    images: torch.Tensor, captions: torch.Tensor, bboxes: torch.Tensor
) -> None:
    """
    Debug function to print data shapes for troubleshooting.
    """
    print("=== Data Shape Debug ===")
    print(f"Images shape: {images.shape}")
    print(f"Captions shape: {captions.shape}")
    print(f"Bboxes shape: {bboxes.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Captions dtype: {captions.dtype}")
    print(f"Bboxes dtype: {bboxes.dtype}")
    print(f"Images device: {images.device}")
    print(f"Captions device: {captions.device}")
    print(f"Bboxes device: {bboxes.device}")
    print("=====================")
