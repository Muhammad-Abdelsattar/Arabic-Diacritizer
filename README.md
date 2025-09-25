# End-to-End Arabic Diacritizer

This repository provides a complete, end-to-end pipeline for training, evaluating, and deploying high-performance Arabic diacritization models. The entire workflow is managed through a clean command-line interface (CLI) and YAML configuration files, making it easy to go from raw text to a production-ready inference engine.

### ✨ Core Philosophy: Efficiency and Effectiveness

The goal of this project is to strike a crucial balance between **model performance** and **computational efficiency**. While massive, complex models may achieve state-of-the-art results on benchmarks, they are often impractical for real-world deployment, especailly on device (edge deployment).

This project prioritizes creating models that are:

- **Lightweight:** Small in size and memory footprint.
- **Fast:** Optimized for quick inference, even on CPU.
- **Effective:** Maintain high accuracy for practical use cases.

We achieve this by focusing on efficient architectures (like small BiLSTMs and compact Transformers) and providing a seamless export path to the highly optimized ONNX Runtime.

---

### 🚀 Key Features

- **Hugging Face Hub Integration:** Get started instantly. The inference package automatically downloads and caches pre-trained models (`small`, `medium`, `large`) for you.
- **Complete CLI Workflow:** Manage the entire lifecycle from the command line:
    1.  `preprocess`: Clean and segment raw text corpora.
    2.  `train`: Train, fine-tune, or resume runs with a single command.
    3.  `evaluate`: Test model performance on a held-out set.
    4.  `export`: Convert a PyTorch checkpoint into a deployment-ready ONNX model.
- **Modular Model Factory:** Easily switch between different architectures (`BiLSTM`, `TransformerEncoder`) by changing a single line in a config file.
- **Built on PyTorch Lightning:** Leverages a robust, industry-standard training framework.
- **Lightweight Inference Engine:** The inference package is completely decoupled from PyTorch. It runs on **ONNX Runtime**, making it fast, portable, and easy to integrate into any application.

---

### ⚡ Quick Start

Get started in minutes, whether you want to use a model or train one.

#### 1. Using a Pre-trained Model (Inference)

To use an exported model, you only need the lightweight `inference` package.

First, clone the repository and install the dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/muhammad-abdelsattar/arabic-diacritizer.git
cd arabic-diacritizer

# 2. Install the inference package and its dependencies
pip install ./inference huggingface-hub onnxruntime 

```

Then, you can use the `Diacritizer` class to diacritize your text. Make sure to load the model from the directory containing `model.onnx` and `vocab.json`.

```python

from arabic_diacritizer_inference import Diacritizer

# 1. Load the model from the directory containing model.onnx and vocab.json
# This directory is created by the `export` command.
diacritizer = Diacritizer("path/to/exported_model_dir")

# 2. Diacritize your text
text = "مرحبا بالعالم"
diacritized_text = diacritizer.diacritize(text)

print(diacritized_text)
```

#### 2. Training a New Model

To train a model, you'll need the full repository and the training dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/muhammad-abdelsattar/arabic-diacritizer.git
cd arabic-diacritizer

# 2. Install the required dependencies
# (Ensure you have a virtual environment activated)
pip install -r requirements.txt

# 3. Start a training run using a configuration file
# This command trains a Transformer model using the specified config.
python scripts/run.py train --experiment configs/lstm.yaml
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

The repository is organized into several key packages to maintain a clean separation of concerns:

```
├── docs/                 # Detailed documentation guides
├── arabic_diacritizer/   # The core training library (data loaders, models, trainer)
├── common/               # Shared text processing utilities (tokenizer, cleaners)
├── configs/              # YAML configuration files for experiments
├── inference/            # The lightweight, PyTorch-free inference package
└── scripts/              # The command-line interface (CLI) entry points
```

---

### License

This project is licensed under the [MIT License](LICENSE).
