#!/bin/bash

# Default values
MODEL_ID="microsoft/Florence-2-large"
MAX_NEW_TOKENS=1024
NUM_BEAMS=3
INPUT_IMAGE=None  # Default to None
SAVE_RESULTS=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num-beams)
            NUM_BEAMS="$2"
            shift 2
            ;;
        --input-image)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --save-results)
            SAVE_RESULTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
echo "Running Image Captioning with:"
echo "  Model ID: $MODEL_ID"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Num Beams: $NUM_BEAMS"
echo "  Input Image: $INPUT_IMAGE"
echo "  Save Results: $SAVE_RESULTS"

python3 main.py --model_id "$MODEL_ID" --max_new_tokens "$MAX_NEW_TOKENS" --num_beams "$NUM_BEAMS" --input_image_filepath "$INPUT_IMAGE" --save_results "$SAVE_RESULTS"
