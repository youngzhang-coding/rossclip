import argparse
import yaml
import torch

from eval.reconstruction import eval_image_reconstruction


def main():
    parser = argparse.ArgumentParser(description="Evaluate image reconstruction")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save reconstructed image")
    parser.add_argument("--cpkt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--detector_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--spacy_language_model_id", type=str, default="en_core_web_sm")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--weights_only", type=str, choices=['true', 'false', 'auto'], default='auto',
                        help="Whether to use weights_only loading")

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    cfg['device'] = args.device

    if args.weights_only == 'true':
        weights_only = True
    elif args.weights_only == 'false':
        weights_only = False
    else:
        weights_only = None 

    try:
        eval_image_reconstruction(
            input_path=args.input_path,
            output_path=args.output_path,
            detector_id=args.detector_id,
            spacy_language_model_id=args.spacy_language_model_id,
            cpkt_path=args.cpkt_path,
            cfg=cfg,
            weights_only=weights_only,
            device=args.device
        )

        print(f"Reconstruction completed. Output saved to {args.output_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()