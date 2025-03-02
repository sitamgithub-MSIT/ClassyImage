# Necessary imports
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from src.siglip.classifier import ZeroShotImageClassification


# Examples to display in the interface
examples = [
    [
        "images/baklava.png",
        "dessert on a plate, a serving of baklava, a plate and spoon",
    ],
    [
        "images/beignets.png",
        "a dog, a cat, a donut, a beignet",
    ],
    [
        "images/cat.png",
        "two sleeping cats, two cats playing, three cats laying down",
    ],
]

# Title and description and article for the interface
title = "Zero Shot Image Classification"
description = "Classify image using zero-shot classification with SigLIP 2 model! Provide an image input and a list of candidate labels separated by commas. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2502.14786' target='_blank'>SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features</a> | <a href='https://huggingface.co/google/siglip2-so400m-patch16-naflex' target='_blank'>Model Page</a></p>"


# Launch the interface
demo = gr.Interface(
    fn=ZeroShotImageClassification,
    inputs=[
        gr.Image(type="pil", label="Input", placeholder="Enter image to classify"),
        gr.Textbox(
            label="Candidate Labels",
            placeholder="Enter candidate labels separated by commas",
        ),
    ],
    outputs=gr.Label(label="Classification", num_top_classes=3),
    title=title,
    description=description,
    article=article,
    examples=examples,
    cache_examples=True,
    cache_mode="lazy",
    theme="Soft",
    flagging_mode="never",
)
demo.launch(debug=False)
