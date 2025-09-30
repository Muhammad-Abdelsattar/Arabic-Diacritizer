import gradio as gr
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diacritizer import Diacritizer

# Use a dictionary as a simple cache to store initialized models.
# This prevents reloading models from disk on every request.
MODEL_CACHE = {}

# Metadata for the UI to display.
MODEL_INFO = {
    "small": "**Small Model**: **Size:** 4 MB",
    "medium": "**Medium Model**: **Size:** 15.5 MB",
}


def get_model(size: str):
    """Lazily loads and caches a model to conserve memory."""
    if size not in MODEL_CACHE:
        print(f"Loading model '{size}' for the first time...")
        try:
            MODEL_CACHE[size] = Diacritizer(size=size)
        except Exception as e:
            raise gr.Error(f"Failed to load the '{size}' model. Error: {e}")
    return MODEL_CACHE[size]


def diacritize_text(text, model_size):
    """
    Takes raw text and a model size, returns the diacritized version and inference time.
    """
    if not text.strip():
        return "", "0.00s"

    diacritizer = get_model(model_size)

    start_time = time.time()
    diacritized_text = diacritizer.diacritize(text)
    end_time = time.time()

    inference_time = f"{end_time - start_time:.3f}s"
    return diacritized_text, inference_time


DESCRIPTION = """
# ⚡ End-to-End Arabic Diacritizer
A lightweight and efficient model for automatic Arabic diacritization.
Select a model, enter some text, and see it in action. For more details, visit the
[GitHub repository](https://github.com/muhammad-abdelsattar/arabic-diacritizer).
"""

# Pre-load the default model for a faster first-time user experience.
print("Pre-loading default 'medium' model...")
get_model("medium")
print("Default model loaded successfully.")

# Use the default theme which is optimized for both light and dark modes.
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row(elem_id="controls-row"):
        model_selector = gr.Radio(
            ["medium", "small"],
            label="Model Size",
            value="medium",
            info="Choose the model to use for diacritization.",
        )
        inference_time_output = gr.Textbox(
            label="Inference Time", interactive=False, max_lines=1
        )

    info_display = gr.Markdown(MODEL_INFO["medium"], elem_id="info-display")

    input_text = gr.Textbox(
        label="Input Text",
        placeholder="اكتب جملة عربية غير مشكولة هنا...",
        lines=6,
        rtl=True,
    )

    submit_button = gr.Button("Diacritize", variant="primary")

    output_text = gr.Textbox(
        label="Diacritized Text", lines=6, rtl=True, interactive=False
    )

    gr.Examples(
        [
            ["أعلنت الشركة عن نتائجها المالية للربع الأول من العام."],
            ["إن مع العسر يسرا."],
            ["هل يمكن للذكاء الاصطناعي أن يكون مبدعا؟"],
            ["كان المتنبي شاعرا عظيما في العصر العباسي."],
        ],
        inputs=input_text,
        label="Examples",
    )

    submit_button.click(
        fn=diacritize_text,
        inputs=[input_text, model_selector],
        outputs=[output_text, inference_time_output],
    )

    model_selector.change(
        fn=lambda size: MODEL_INFO[size], inputs=model_selector, outputs=info_display
    )

if __name__ == "__main__":
    demo.launch()
