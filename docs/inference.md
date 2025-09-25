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
from arabic_diacritizer_inference import Diacritizer

# The first time you run this, it will download the model.
# Subsequent runs will be instant as it uses the local cache.
diacritizer = Diacritizer()

diacritized_text = diacritizer.diacritize("مرحبا بالعالم")
print(diacritized_text)
```

### Loading a Specific Model Size

If you need a different balance of speed and accuracy, you can request a specific model size (`small`, `medium`, or `large`).

```python
# Request the 'small' model for maximum speed
diacritizer_small = Diacritizer(size="small")

# Request the 'large' model for the highest accuracy
diacritizer_large = Diacritizer(size="large")
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
# This will always use the model version tagged as 'v1.2'.
# It guarantees that your results are reproducible, even if the 'main' branch is updated.
diacritizer_v1_2 = Diacritizer(size="large", revision="v1.2")
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
gpu_diacritizer = Diacritizer(size="large", use_gpu=True)
```

## 4. How it Works: Caching

The `Diacritizer` is designed to be efficient. The first time you request a model from the Hugging Face Hub, it is downloaded and stored in a central cache directory on your machine.

*   **Speed:** All subsequent initializations of the same model will be instantaneous, as they will read directly from the local cache.
*   **Offline Access:** Once a model is cached, you can use the `Diacritizer` without an internet connection.
```Of course. This is a critical final step. The documentation must reflect the new, powerful, and user-friendly capabilities of the inference package.

Here are the updated versions of `README.md` and `docs/inference.md`.

---

### Updated `README.md`

The main `README.md` is updated to showcase the new, simplified "Quick Start" for inference, immediately highlighting the Hugging Face Hub integration.

```markdown
# End-to-End Arabic Diacritizer

This repository provides a complete, end-to-end pipeline for training, evaluating, and deploying high-performance Arabic diacritization models. The entire workflow is managed through a clean command-line interface (CLI) and YAML configuration files, making it easy to go from raw text to a production-ready inference engine.

### ✨ Core Philosophy: Efficiency and Effectiveness

The goal of this project is to strike a crucial balance between **model performance** and **computational efficiency**. While massive, complex models may achieve state-of-the-art results on academic benchmarks, they are often impractical for real-world deployment.

This project prioritizes creating models that are:
*   **Lightweight:** Small in size and memory footprint.
*   **Fast:** Optimized for quick inference, even on CPU.
*   **Effective:** Maintain high accuracy for practical use cases.

We achieve this by focusing on efficient architectures (like BiLSTMs and compact Transformers) and providing a seamless export path to the highly optimized ONNX Runtime.

---

### 🚀 Key Features

*   **Hugging Face Hub Integration:** Get started instantly. The inference package automatically downloads and caches pre-trained models (`small`, `medium`, `large`) for you.
*   **Complete CLI Workflow:** Manage the entire lifecycle from the command line:
    1.  `preprocess`: Clean and segment raw text corpora.
    2.  `train`: Train, fine-tune, or resume runs with a single command.
    3.  `evaluate`: Test model performance on a held-out set.
    4.  `export`: Convert a PyTorch checkpoint into a deployment-ready ONNX model.
*   **Modular Model Factory:** Easily switch between different architectures (`BiLSTM`, `TransformerEncoder`) by changing a single line in a config file.
*   **Built on PyTorch Lightning:** Leverages a robust, industry-standard training framework.
*   **Lightweight Inference Engine:** The inference package is completely decoupled from PyTorch. It runs on **ONNX Runtime**, making it fast, portable, and easy to integrate into any application.

---

### ⚡ Quick Start

Get started in minutes, whether you want to use a model or train one.

#### 1. Using a Pre-trained Model (Inference)

The inference package automatically downloads and caches a pre-trained model for you.

```python
# First, install the inference package and its dependencies
# pip install ./inference huggingface-hub onnxruntime numpy

from arabic_diacritizer_inference import Diacritizer

# 1. Initialize the diacritizer. The default 'medium' model will be
#    downloaded from the Hugging Face Hub and cached locally.
#    Subsequent initializations will be instant.
diacritizer = Diacritizer()

# 2. Diacritize your text
text = "مرحبا بالعالم"
diacritized_text = diacritizer.diacritize(text)

print(diacritized_text)
# Expected output: مَرْحَبًا بِالْعَالَمِ```

#### 2. Training a New Model

To train a model, you'll need the full repository and the training dependencies.

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/muhammad-abdelsattar/arabic-diacritizer.git
cd arabic-diacritizer
pip install -r requirements.txt
pip install -e ./common

# 2. Start a training run using a configuration file
python scripts/run.py train --experiment configs/transformer.yaml
```

---

### 📚 Full Documentation

This README provides a high-level overview. For detailed instructions, please refer to the guides below.

> **Looking to use the model in your application?**
>
> See the **[➡️ Inference Guide](./docs/inference.md)** for installation and detailed usage instructions.

> **Want to train, fine-tune, or export your own model?**
>
> See the **[➡️ Training & Development Guide](./docs/training.md)** for a complete walkthrough of the data preparation, configuration, training, and export process.

---

### 📂 Project Structure
(This section remains the same)
```

---

### Updated `docs/inference.md`

This is a complete rewrite of the inference guide. It's now structured to introduce the new features logically, from the simplest use case to the most advanced.

```markdown
# Inference Guide

This guide covers everything you need to know to use the Arabic Diacritizer in your own Python applications.

The inference engine is designed to be powerful yet easy to use. It can automatically download and cache pre-trained models from the Hugging Face Hub, or load a local model you have trained yourself. The engine is lightweight and runs on the highly optimized ONNX Runtime.

## 1. Installation

You need the `inference/` package and its dependencies. The `huggingface-hub` library is now required for automatic model downloading.

```bash
# Run this from the root directory of the project
pip install ./inference huggingface-hub onnxruntime numpy
```
You now have everything needed to run inference.

## 2. Basic Usage

These are the most common ways to use the `Diacritizer`.

### Loading the Default Model

For most use cases, you can simply initialize the `Diacritizer` without any arguments. It will automatically download and cache the default `medium`-sized model.

```python
from arabic_diacritizer_inference import Diacritizer

# The first time you run this, it will download the model.
# Subsequent runs will be instant as it uses the local cache.
diacritizer = Diacritizer()

diacritized_text = diacritizer.diacritize("مرحبا بالعالم")
print(diacritized_text)
```

### Loading a Specific Model Size

If you need a different balance of speed and accuracy, you can request a specific model size (`small`, `medium`, or `large`).

```python
# Request the 'small' model for maximum speed
diacritizer_small = Diacritizer(size="small")

# Request the 'medium' model for a balance of speed and accuracy
diacritizer_large = Diacritizer(size="medium")
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
# This will always use the model version tagged as 'v1.2'.
# It guarantees that your results are reproducible, even if the 'main' branch is updated.
diacritizer_v1_2 = Diacritizer(size="large", revision="v1.2")
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
gpu_diacritizer = Diacritizer(size="large", use_gpu=True)
```

## 4. How it Works: Caching

The `Diacritizer` is designed to be efficient. The first time you request a model from the Hugging Face Hub, it is downloaded and stored in a central cache directory on your machine.

*   **Speed:** All subsequent initializations of the same model will be instantaneous, as they will read directly from the local cache.
*   **Offline Access:** Once a model is cached, you can use the `Diacritizer` without an internet connection.

