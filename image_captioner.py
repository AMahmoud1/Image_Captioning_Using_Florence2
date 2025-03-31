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

