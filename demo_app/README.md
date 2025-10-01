# Demo App

This directory contains a self-contained Gradio application for demonstrating the Arabic Diacritizer model.

## Purpose

The `app.py` script creates a user-friendly web interface that allows users to input undiacritized Arabic text and receive the diacritized output from the model in real-time. This demo is designed to be deployed directly as a **Hugging Face Space**.

## Running Locally

1.  **Install Dependencies from Project Root:**
    Before running the demo, you must install its dependencies and the project's local packages from the **root directory** of the repository.

    ```bash
    # From the project's root directory:

    # 1. Install the shared common library
    pip install ./common

    # 2. Install the inference engine
    pip install ./inference

    # 3. Install Gradio and other demo-specific requirements
    pip install -r demo_app/requirements.txt
    ```

2.  **Run the App:**
    Once the dependencies are installed, you can run the Gradio application.

    ```bash
    python demo_app/app.py
    ```

    A local web server will start. You can access the demo in your browser at the URL provided in your terminal (usually `http://127.0.0.1:7860`).
