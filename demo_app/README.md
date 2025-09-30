# Hugging Face Spaces Demo

This directory contains a self-contained Gradio application for demonstrating the Arabic Diacritizer model.

## Purpose

The `app.py` script creates a user-friendly web interface that allows users to input undiacritized Arabic text and receive the diacritized output from the model in real-time.

This demo is designed to be deployed directly as a **Hugging Face Space**.

## Running Locally

1.  **Install Dependencies:**
    Make sure you have installed the project's core dependencies from the root directory, then install the Gradio-specific requirements.

    ```bash
    # From the project root directory
    pip install -r demo_app/requirements.txt
    pip install ./common
    pip install ./inference
    ```

2.  **Run the App:**
    Execute the `app.py` script.

    ```bash
    python demo_app/app.py
    ```

    A local web server will start, and you can access the demo in your browser at `http://127.0.0.1:7860`.
