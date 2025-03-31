import argparse

from image_captioner import ImageCaptioner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image Captioning")

    parser.add_argument(
        "--model_id", type=str, default="microsoft/Florence-2-large", help="Model ID"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--num_beams", type=int, default=3, help="Beam search width")
    parser.add_argument(
        "--input_image_filepath",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--save_results", type=bool, default=True, help="Whether to save the results"
    )

    args = parser.parse_args()

    # Create Instance of Image Captioner
    image_captioner = ImageCaptioner(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    # Infer the Image Captioner
    prompt = "<DENSE_REGION_CAPTION>"

    image_captioner.infer(
        input_image_filepath=(
            None if args.input_image_filepath == "None" else args.input_image_filepath
        ),
        prompt=prompt,
        save_results=args.save_results,
    )
