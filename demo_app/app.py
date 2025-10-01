import gradio as gr
import time
from diacritizer import Diacritizer, ModelNotFound

MODEL_INFO = {
    "bilstm": {
        "display_name": "BiLSTM",
        "models": {
            "medium": {
                "size": "4 MB",
                "details": "Balanced speed and accuracy.",
            },
            "large": {
                "size": "15.5 MB",
                "details": "Highest accuracy model.",
            },
        },
    },
    "bigru": {
        "display_name": "BiGRU",
        "models": {
            "medium": {
                "size": "3.8 MB",
                "details": "Slightly faster than BiLSTM with comparable accuracy.",
            },
            "large": {
                "size": "14.9 MB",
                "details": "High accuracy alternative to the BiLSTM model.",
            },
        },
    },
}

MODEL_CACHE = {}


def get_model(architecture: str, size: str, progress=gr.Progress()):
    """
    Lazily loads and caches a Diacritizer model.
    Includes user feedback via gr.Progress to show loading status.
    """
    model_key = f"{architecture}/{size}"
    if model_key not in MODEL_CACHE:
        progress(0.5, desc=f"Loading {architecture}/{size} model...")
        try:
            MODEL_CACHE[model_key] = Diacritizer(architecture=architecture, size=size)
        except ModelNotFound:
            raise gr.Error(
                f"The requested model ({model_key}) was not found on the Hugging Face Hub."
            )
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred while loading the model: {e}")
    return MODEL_CACHE[model_key]


def diacritize_text(text: str, architecture: str, size: str, progress=gr.Progress()):
    """
    Main function to diacritize text, now with progress tracking.
    """
    if not text or not text.strip():
        return "", "0.000s", "Please enter some text to diacritize."

    progress(0, desc="Loading model...")
    diacritizer = get_model(architecture, size, progress)

    progress(0.8, desc="Diacritizing text...")
    start_time = time.time()
    diacritized_text = diacritizer.diacritize(text)
    end_time = time.time()

    inference_time = f"{end_time - start_time:.3f}s"

    # Update the info text with the final result details
    model_details = MODEL_INFO[architecture]["models"][size]["details"]
    final_info_text = f"**Model:** {architecture}/{size} | **Size:** {MODEL_INFO[architecture]['models'][size]['size']} | {model_details}"

    return diacritized_text, inference_time, final_info_text


def update_available_sizes(architecture: str):
    """Callback to update the size choices when the architecture changes."""
    available_sizes = list(MODEL_INFO[architecture]["models"].keys())
    # Return a new Radio component with updated choices and a default value
    return gr.Radio(
        choices=available_sizes,
        value=available_sizes[0],  # Default to the first available size
        label="Model Size",
        info="Select the model size.",
    )


theme = gr.themes.Soft(
    primary_hue="zinc",
    secondary_hue="blue",
    neutral_hue="slate",
    font=(gr.themes.GoogleFont("Noto Sans"), gr.themes.GoogleFont("Noto Sans Arabic")),
).set(
    body_background_fill_dark="#111827"  # A slightly off-black for dark mode
)

DESCRIPTION = """
# ⚡ End-to-End Arabic Diacritizer
A lightweight and efficient model for automatic Arabic diacritization.
Select an architecture and size, enter some text, and see it in action. For more details, visit the
[GitHub repository](https://github.com/muhammad-abdelsattar/arabic-diacritizer).
"""

with gr.Blocks(theme=theme, css=".footer {display: none !important}") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                arch_selector = gr.Radio(
                    choices=[
                        (info["display_name"], arch)
                        for arch, info in MODEL_INFO.items()
                    ],
                    label="Model Architecture",
                    value="bilstm",
                    info="Select the model architecture.",
                )
                model_selector = gr.Radio(
                    choices=["medium", "large"],
                    label="Model Size",
                    value="medium",
                    info="Select the model size.",
                )
            info_display = gr.Markdown(
                "**Model:** bilstm/medium | **Size:** 4 MB | Balanced speed and accuracy. (Formerly 'small')",
                elem_id="info-display",
            )

        with gr.Column(scale=1):
            inference_time_output = gr.Textbox(
                label="Inference Time", interactive=False, max_lines=1
            )

    with gr.Row(equal_height=True):
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text (Undiacritized)",
                placeholder="اكتب جملة عربية غير مشكولة هنا...",
                lines=8,
                rtl=True,
            )
        with gr.Column():
            output_text = gr.Textbox(
                label="Output Text (Diacritized)",
                lines=8,
                rtl=True,
                interactive=False,
            )

    submit_button = gr.Button("Diacritize ✨", variant="primary")

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
        inputs=[input_text, arch_selector, model_selector],
        outputs=[output_text, inference_time_output, info_display],
    )

    # When architecture changes, update the available sizes
    arch_selector.change(
        fn=update_available_sizes, inputs=arch_selector, outputs=model_selector
    )


if __name__ == "__main__":
    # Pre-load the default model for a faster first-time user experience
    print("Pre-loading default 'bilstm/medium' model...")
    get_model(architecture="bilstm", size="medium")
    print("Default model loaded successfully.")

    demo.launch()
