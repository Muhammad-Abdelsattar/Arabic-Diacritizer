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

- **Live Gradio Demo:** Try the model instantly in your browser via our [**Hugging Face Space**](https://huggingface.co/spaces/Muhammad7777/Arabic-Diacritizer).
- **Hugging Face Hub Integration:** Get started instantly. The inference package automatically downloads and caches pre-trained models by architecture (`bilstm`, `bigru`) and size (`snall`, `medium`, `large`).
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
from diacritizer import Diacritizer

# 1. Initialize the diacritizer. The default 'medium' model will be
#    downloaded from the Hugging Face Hub and cached locally.
#    Subsequent initializations will be instant.
diacritizer = Diacritizer(architecture="bilstm", size="medium")

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
> See the **[➡️Inference Guide](./docs/inference.md)** for installation and detailed usage instructions using the lightweight inference library.

> **Want to see performance numbers and comparisons?**
>
> See the **[➡️Benchmarks & Performance](./docs/benchmarks.md)** page for DER/WER scores, model sizes, and SOTA comparisons.

> **Want to train, fine-tune, or export your own model?**
>
> See the **[➡️Training & Development Guide](./docs/training.md)** for a complete walkthrough of the data preparation, configuration, training, and export process.

---

### 📂 Project Structure

The repository is organized into several key packages to maintain a clean separation of concerns:

```
├── docs/                 # Detailed documentation guides
├── demo_app/             # Source code for the live Gradio demo
├── arabic_diacritizer/   # The core training library (data loaders, models, trainer)
├── common/               # Shared text processing utilities (tokenizer, cleaners)
├── configs/              # YAML configuration files for experiments
├── inference/            # The lightweight, PyTorch-free inference package
├── scripts/              # The command-line interface (CLI) entry points
└── deploy_space.sh       # Automation script to deploy the Gradio demo to HF Spaces
```

---

### Acknowledgements

This project's architecture and training methodologies have been greatly inspired by the work of the open-source Arabic NLP community.

> A special acknowledgement and thanks goes to the **[Hareef](https://github.com/mush42/hareef)** repository. Studying its `hareef/sarf` model was an invaluable learning experience. Specifically, the innovative idea of training with partially diacritized inputs (what we call a "hint-based" mechanism) was a direct inspiration for some of the advanced models in this project. This technique has proven to be highly effective for improving the model's robustness and contextual understanding.

> The [Sadeed Benchmark](https://huggingface.co/datasets/Misraj/SadeedDiac-25) for the evaluation benchmark.

---

### License

This project is licensed under the [MIT License](LICENSE).
