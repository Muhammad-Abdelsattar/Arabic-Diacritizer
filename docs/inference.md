# Inference Guide

This guide will show you how to use a trained and exported Arabic Diacritizer model in your own Python applications.

The inference engine is designed to be powerful yet easy to use. It can automatically download and cache pre-trained models from the Hugging Face Hub, or load a local model you have trained yourself. The engine is lightweight and runs on the highly optimized ONNX Runtime.

## 1. Installation

You need the `inference/` package and its dependencies: `onnxruntime`. The `huggingface-hub` library is now required for automatic model downloading.

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
    - `huggingface-hub`: Used to download the models and artifacts from the Hugging Face Hub.

It's that simple. You now have everything you need to run inference.

## 2. Basic Usage

These are the most common ways to use the `Diacritizer`.

### Loading the Default Model

For most use cases, you can simply initialize the `Diacritizer` without any arguments. It will automatically download and cache the default `medium`-sized model.

```python
from diacritizer import Diacritizer

# The first time you run this, it will download the model.
# Subsequent runs will be instant as it uses the local cache.
diacritizer_bilstm = Diacritizer(architecture="bilstm", size="medium")

diacritized_text = diacritizer.diacritize("مرحبا بالعالم")
print(diacritized_text)
```

### Loading a Specific Model Size

If you need a different balance of speed and accuracy, you can request a specific model size (`small`, `medium`, or `large`).

```python
# Request the 'small' model for maximum speed
diacritizer_small = Diacritizer(architecture="bilstm", size="small")

# Request the 'large' model for the highest accuracy
diacritizer_large = Diacritizer(architecture="bigru", size="large")
```

## 3. Advanced Usage

The `Diacritizer` provides additional options for fine-grained control over model loading, which is crucial for custom models and production environments.

### Using a Local Model

If you have trained or fine-tuned your own model, you can load it directly from a local directory. The package will automatically detect that you've provided a local path.

The directory must contain `model.onnx` and `vocab.json`.

```python
# The model_identifier is now a local file path.
# The package will not contact the Hugging Face Hub.
local_diacritizer = Diacritizer(model_identifier="./path/to/my-finetuned-model/")
```

### Model Versioning for Reproducibility

For production systems or scientific research, it's critical to pin your code to a specific model version. You can do this with the `revision` parameter, which corresponds to a Git tag, branch, or commit hash in the Hugging Face Hub repository.

```python
# It guarantees that your results are reproducible, even if the 'main' branch is updated.
diacritizer_v1_2 = Diacritizer(architecture="bilstm", size="medium", revision="v1.2")
```

### Force-Syncing to the Latest Version

By default, the package uses its local cache. If you always need the absolute latest version of a model from the `main` branch, you can force a sync.

```python
# Bypasses the cache and checks the Hub for a newer version.
# Note: This requires an internet connection every time it's initialized.
diacritizer_latest = Diacritizer(force_sync=True)
```

### GPU Inference

For the best performance, you can run inference on a GPU. You must first install the GPU-enabled version of ONNX Runtime (`pip install onnxruntime-gpu`).

```python
# The use_gpu flag tells the predictor to use the CUDA execution provider.
gpu_diacritizer = Diacritizer(architecture="bilstm", size="medium", use_gpu=True)
```

## 4. How it Works: Caching

The `Diacritizer` is designed to be efficient. The first time you request a model from the Hugging Face Hub, it is downloaded and stored in a central cache directory on your machine.

- **Speed:** All subsequent initializations of the same model will be instantaneous, as they will read directly from the local cache.
- **Offline Access:** Once a model is cached, you can use the `Diacritizer` without an internet connection.

