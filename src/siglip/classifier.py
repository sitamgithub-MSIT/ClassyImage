# Necessary imports
import sys
from typing import Dict
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import gradio as gr

# Local imports
from src.logger import logging
from src.exception import CustomExceptionHandling


# Load the zero-shot image classification model
model_id = "google/siglip2-so400m-patch16-naflex"
model = AutoModel.from_pretrained(model_id).eval().to("cpu")
processor = AutoProcessor.from_pretrained(model_id)


def ZeroShotImageClassification(
    image_input: Image.Image, candidate_labels: str
) -> Dict[str, float]:
    """
    Performs zero-shot classification on the given image input and candidate labels.

    Args:
        - image_input: The input image to classify.
        - candidate_labels: A comma-separated string of candidate labels.

    Returns:
        Dictionary containing label-score pairs.
    """
    try:
        # Check if the input and candidate labels are valid
        if not image_input or not candidate_labels:
            gr.Warning("Please provide valid input and candidate labels")

        # Split and clean the candidate labels
        labels = [label.strip() for label in candidate_labels.split(",")]

        # Log the classification attempt
        logging.info(f"Attempting classification with {len(labels)} labels")

        # Perform zero-shot image classification
        inputs = processor(
            text=labels,
            images=image_input,
            return_tensors="pt",
            max_num_patches=256,
        ).to("cpu")
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)

        # Return the classification results
        logging.info("Classification completed successfully")
        return {labels[i]: probs[0][i] for i in range(len(labels))}

    # Handle exceptions that may occur during the process
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
