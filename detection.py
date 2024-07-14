import gradio as gr
from transformers import pipeline

from detection_utils import render_results_in_image, summarize_predictions_natural_language

obj_detector = pipeline(
    task = "object-detection",
    model = "facebook/detr-resnet-50"
)

def get_pipeline_prediction(pil_image):
    pipeline_output = obj_detector(pil_image)
    processed_image = render_results_in_image(
        pil_image, 
        pipeline_output
    )
    detection_summary = summarize_predictions_natural_language(pipeline_output)
    return processed_image, detection_summary

detection_interface = gr.Interface(
    fn = get_pipeline_prediction,
    inputs = gr.Image(
        label = "Input Image",
        type = 'pil'
    ),
    outputs = [gr.Image(
    label = "Output image with predicted instances",
    type = 'pil'
    ),
    gr.Textbox(label="Detection Summary")],
    allow_flagging = 'never'
    
)

# Add Markdown content
markdown_content_detection = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Object Detection with Summary</h1>
        <h3 style='color: #4682B4;'>Model: facebook/detr-resnet-50</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
detection_with_markdown = gr.Blocks()
with detection_with_markdown:
    markdown_content_detection.render()
    detection_interface.render()