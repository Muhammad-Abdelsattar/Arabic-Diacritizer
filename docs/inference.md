# Inference Guide

This guide will show you how to use a trained and exported Arabic Diacritizer model in your own Python applications.

The inference engine is designed to be **lightweight and efficient**. It is completely decoupled from PyTorch and runs on the highly optimized ONNX Runtime, making it easy to install and deploy with minimal dependencies.

## 1. Installation

You only need the `inference/` package and its two main dependencies: `onnxruntime` and `numpy`.

1.  **Clone and Navigate to the project's root directory.**

```bash
# 1. Clone the repository
git clone https://github.com/muhammad-abdelsattar/arabic-diacritizer.git
cd arabic-diacritizer
```

2.  **Install the necessary packages using pip.**
    This single command installs the local `arabic_diacritizer_inference` package along with its required libraries.

    ```bash
    pip install ./inference
    ```

    - `./inference`: Installs the local package containing the `Diacritizer` class.
    - `onnxruntime`: The high-performance engine that runs the model.
    - `numpy`: Used for numerical operations.

It's that simple. You now have everything you need to run inference.

## 2. Required Model Files

To load a model, the `Diacritizer` class needs a directory containing the following two files:

1.  `model.onnx`
2.  `vocab.json`

These files are the standard output of the `export` command from the main training pipeline. Your model directory should look like this:

```
my_exported_model/
├── model.onnx
└── vocab.json
```

## 3. Basic Usage

Using the model is straightforward. The following example shows how to load the diacritizer and process a sentence.

```python
from arabic_diacritizer_inference import Diacritizer
from arabic_diacritizer_inference import InvalidInputError, ModelNotFound

# Define the path to your exported model directory.
model_dir = "path/to/my_exported_model/"

try:
    # Initialize the Diacritizer object.
    # This will load the ONNX model and the vocabulary.
    diacritizer = Diacritizer(model_dir)

    # Provide the text you want to diacritize.
    text = "مرحبا بالعالم"

    # Call the `diacritize` method.
    diacritized_text = diacritizer.diacritize(text)

    print(f"Original: {text}")
    print(f"Diacritized: {diacritized_text}")

except ModelNotFound:
    print(f"Error: Model files not found in '{model_dir}'.")
    print("Please ensure 'model.onnx' and 'vocab.json' are present.")

except InvalidInputError as e:
    print(f"Error: Invalid input provided. {e}")

```


## 4. Advanced Topics

#### GPU Inference

For significantly faster performance on compatible hardware, you can instruct the diacritizer to use your GPU.

**Prerequisite:** You must have the GPU-enabled version of ONNX Runtime installed.

```bash
pip install onnxruntime-gpu
```

You also need the appropriate NVIDIA CUDA drivers installed on your system.

**Usage:**
Simply pass the `use_gpu=True` flag when initializing the `Diacritizer`.

```python
# Initialize the diacritizer to run on the GPU.
diacritizer = Diacritizer(model_dir, use_gpu=True)

# The rest of your code remains the same.
diacritized_text = diacritizer.diacritize("...")
```

#### Handling Long Texts

You do not need to worry about the length of the input text. The `diacritize` method is designed to handle everything from single words to entire articles seamlessly.

Internally, it automatically detects long inputs and breaks them down into smaller segments that the model can process. It then diacritizes each segment and stitches the results back together, returning a single, fully-diacritized string.
