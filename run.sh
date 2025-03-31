#!/bin/bash

# Name of the Conda virtual environment
VENV_NAME="image_captioning_venv"

# Check if the Conda environment exists
env_exists=$(conda env list | grep "^$VENV_NAME")

if [ -z "$env_exists" ]; then
    echo "Conda environment '$VENV_NAME' does not exist. Creating..."
    conda create -y -n $VENV_NAME python=3.9
else
    echo "Conda environment '$VENV_NAME' already exists."
fi

# Activate the Conda environment
source activate $VENV_NAME || conda activate $VENV_NAME

# Install required packages
pip install transformers==4.49.0 ultralytics einops timm

# Run The Application
echo "Image Captioning Application is Starting..."
python3 main.py
echo "Image Captioning Application has Finished"