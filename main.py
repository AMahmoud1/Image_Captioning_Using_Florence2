from image_captioner import ImageCaptioner


if __name__ == "__main__":
    # Define Arguments
    # TODO: To be moved to run script
    model_id = "microsoft/Florence-2-large"
    max_new_tokens=1024  # Maximum number of tokens to generate
    num_beams=3  # Beam search width for better predictions
    input_image_filepath = None
    save_results = True

    # Create Instance of Image Captioner
    image_captioner = ImageCaptioner(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    # Infer the Image Captioner
    prompt = "<DENSE_REGION_CAPTION>"

    image_captioner.infer(
        input_image_filepath=input_image_filepath,
        prompt=prompt,
        save_results=save_results,
    )