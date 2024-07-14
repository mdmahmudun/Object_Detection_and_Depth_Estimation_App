import torch
import numpy as np
import gradio as gr
from transformers import pipeline
from PIL import Image


depth_estimator = pipeline(task = 'depth-estimation',
                           model = 'Intel/dpt-hybrid-midas')

def launch(input_image):
    out = depth_estimator(input_image)

    # resize the prediction
    prediction = torch.nn.functional.interpolate(
        out["predicted_depth"].unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # normalize the prediction
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

depth_interface = gr.Interface(launch, 
                     inputs=gr.Image(type='pil', label = "Input Image"), 
                     outputs=gr.Image(type='pil', label = "Depth Estimation"),
                              allow_flagging = 'never')

# Add Markdown content
markdown_content_depth_estimation = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Image Depth Estimation</h1>
        <h3 style='color: #4682B4;'>Model: Intel/dpt-hybrid-midas</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
depth_estimation_with_markdown = gr.Blocks()
with depth_estimation_with_markdown:
    markdown_content_depth_estimation.render()
    depth_interface.render()
