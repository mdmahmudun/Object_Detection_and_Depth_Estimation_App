import gradio as gr
from detection import detection_with_markdown
from depth_estimation import depth_estimation_with_markdown

# Combine both the app
demo = gr.Blocks()
with demo:
    gr.TabbedInterface(
        [detection_with_markdown, depth_estimation_with_markdown],
        ['Object Detection', 'Depth Estimation']
    )


if __name__ == "__main__":
    demo.launch()