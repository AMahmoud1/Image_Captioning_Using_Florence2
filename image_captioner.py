from transformers import AutoModelForCausalLM, AutoProcessor
import random
import cv2
import torch
from PIL import Image
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
import os


class ImageCaptioner:
    def __init__(self, model_id, max_new_tokens, num_beams):
        # Set Class Attributes
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        # Load Model and Processor
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype="auto").eval().cuda()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def infer(self, input_image_filepath, prompt, save_results):
        """
        Performs inference using a given image and task prompt.

        Args:
            filepath (str): Path to image to read.
            prompt (str): The prompt specifying the task for the model.

        Returns:
            str: The model's processed response after inference.
        """
        # Load Image
        image = self.read_image(input_image_filepath)

        # Generate the input data for model processing from the given prompt and image
        inputs = self.processor(
            text=prompt,  # Text input for the model
            images=image,  # Image input for the model
            return_tensors="pt",  # Return PyTorch tensors
        ).to("cuda", torch.float16)  # Move inputs to GPU with float16 precision

        # Generate model predictions (token IDs)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),  # Convert text input IDs to CUDA
            pixel_values=inputs["pixel_values"].cuda(),  # Convert image pixel values to CUDA
            max_new_tokens=self.max_new_tokens,  # Set maximum number of tokens
            early_stopping=False,  # Disable early stopping
            do_sample=False,  # Use deterministic inference
            num_beams=self.num_beams,  # Set beam search width
        )

        # Decode generated token IDs into text
        generated_text = self.processor.batch_decode(
            generated_ids,  # Generated token IDs
            skip_special_tokens=False,  # Retain special tokens in output
        )[0]  # Extract first result from batch

        # Post-process the generated text into a structured response
        parsed_answer = self.processor.post_process_generation(
            generated_text,  # Raw generated text
            task=prompt,  # Task type for post-processing
            image_size=(image.width, image.height),  # Original image dimensions for scaling output
        )[prompt]

        # Save Results
        if save_results:
            annotator = self.annotate(
                image=image,
                results=parsed_answer,
            )
            self.save_image(
                result_image=annotator.result(),
                path="output.jpg"
            )
        return parsed_answer

    def read_image(self, filename):
        """
        Reads an image from a given filename or selects a random image if no filename is provided.

        Args:
            filename (str or None): The path to the image file. If `None`, a random image is used.

        Returns:
            PIL.Image.Image: The loaded image in RGB format.
        """
        if filename is not None:
            image_name = filename
        else:
            # Pick random online image
            image_name = random.choice(["bus.jpg", "zidane.jpg"])

            if not os.path.isfile(image_name):
                # Download the image if not exist
                safe_download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{image_name}")

        # Read the image using OpenCV and convert it into the PIL format
        return Image.fromarray(cv2.cvtColor(cv2.imread(f"{image_name}"), cv2.COLOR_BGR2RGB))
    
    def annotate(self, image, results):
        """
        Annotates an image with bounding boxes and labels using the Ultralytics annotator.

        Args:
            image (numpy.ndarray or PIL.Image.Image): The input image to annotate.
            results (dict): A dictionary containing:
                - "bboxes" (list): List of bounding boxes in [x1, y1, x2, y2] format.
                - "labels" (list): Corresponding labels for each bounding box.

        Returns:
            Annotator: The annotator object containing the annotated image.
        """
        # Plot the results on an image
        annotator = Annotator(image)  # initialize Ultralytics annotator

        for idx, (box, label) in enumerate(zip(results["bboxes"], results["labels"])):
            annotator.box_label(box, label=label, color=colors(idx, True))
        return annotator
    
    def save_image(self, result_image, path):
        """
        Saves an image to the specified file path.

        Converts the given image array into a PIL Image and saves it to disk.

        Args:
            result_image (numpy.ndarray): The image array to be saved.
            path (str): The file path where the image will be saved.

        Returns:
            None
        """
        output_image = Image.fromarray(result_image)  
        output_image.save(path)  # Save the image