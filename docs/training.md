# Training & Development Guide

This document covers everything you need to know to prepare data, train, fine-tune, evaluate, and export your own Arabic diacritization models using this framework.

## 1. Installation & Setup

Before you can start training, you need to set up your environment with the repository and all the necessary dependencies, including PyTorch.

#### Prerequisites

- Python 3.10+
- Git

#### Setup Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/muhammad-abdelsattar/arabic-diacritizer.git
    cd arabic-diacritizer
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    This project's dependencies are listed in `requirements.txt`. This single command will install everything you need, including PyTorch, PyTorch Lightning, and other utilities.
    ```bash
    pip install -r requirements.txt
    # Install the common package which contains the cleaners and tokenizer
    pip install -e ./common
    ```
    You are now ready to start the workflow!

---

## 2. The End-to-End Workflow

The entire machine learning lifecycle is managed through a simple and powerful command-line interface (CLI) powered by `scripts/run.py`. Here is the standard workflow.

### Step 2.1: Data Preparation (`preprocess`)

Your raw text data, likely in large `.txt` files, needs to be cleaned and normalized before training. The `preprocess` command handles this for you.

**What it does:**

- Filters out non-Arabic characters and invalid diacritics.
- Normalizes whitespace.
- Segments very long lines into shorter, manageable sentences to respect the model's maximum input length.

**Usage:**

```bash
python -m scripts.run preprocess \
  --input path/to/your/raw_corpus_1.txt \
  --input path/to/your/raw_corpus_2.txt \
  --output data/processed/my_dataset.txt
```

This command takes one or more input files and produces a single, clean output file, ready for the training data loader.

### Step 2.2: Training a Model (`train`)

This is the core command, offering three primary modes of operation: starting from scratch, resuming a stopped run, and fine-tuning an existing model.

#### A. Training a New Model

To start a new training run, you typically point to an **experiment configuration file**. These files (e.g., `configs/lstm.yaml`, `configs/transformer.yaml`) define the model architecture and hyperparameters.

```bash
# Train a BiLSTM model using the settings in lstm.yaml
python -m scripts.run train --experiment configs/lstm.yaml

# Train a Transformer model using the settings in transformer.yaml
python -m scripts.run train --experiment configs/your_experiment.yaml
```

The trainer will build the model, load the data specified in the config, and start training from epoch 0. Checkpoints and logs will be saved to the `lightning_logs/` directory by default.

#### B. Resuming a Stopped Training Run

If your training was interrupted, you can resume it seamlessly from the last saved checkpoint. The trainer will restore the model weights, the epoch number, and the optimizer state, continuing exactly where it left off.

To resume, simply provide the path to a checkpoint file using the `--ckpt-path` flag.

```bash
python -m scripts.run train --ckpt-path lightning_logs/version_X/checkpoints/last.ckpt
```

**Important:** When resuming, you do not need to specify an `--experiment` file. The trainer will use the exact configuration that was saved inside the checkpoint file to ensure consistency.

#### C. Fine-tuning a Pre-trained Model

Fine-tuning allows you to take a fully trained model and continue training it on a new (or the same) dataset, but with a fresh start. This is useful for adapting a model to a specific domain or experimenting with different learning rates.

**What it does:**

- Loads **only the model weights** from the checkpoint.
- **Resets the trainer state:** training starts from epoch 0 with a new optimizer and learning rate scheduler.

To fine-tune, provide both a checkpoint path and the `--finetune` flag. You will typically also provide a new experiment config to define the new data and optimizer settings.

```bash
python -m scripts.run train \
  --experiment configs/finetune_config.yaml \
  --ckpt-path path/to/pretrained_model.ckpt \
  --finetune # This flag is required when fine-tuning
```

### Step 2.3: Evaluating the Model (`evaluate`)

Once you have a trained checkpoint, you can run it against the test set to get its final performance metrics (DER, F1-Score, etc.).

```bash
python -m scripts.run evaluate path/to/your_best_model.ckpt

```

The evaluation script will load the model, run the test data loader, and print the final metrics to the console.

### Step 2.4: Exporting for Inference (`export`)

The final step is to convert your PyTorch Lightning checkpoint (`.ckpt`) into a format suitable for the lightweight inference engine.

**What it does:**

- Loads the trained model weights.
- Traces the model and exports it to the **ONNX (Open Neural Network Exchange)** format.
- Saves the tokenizer vocabulary.

**Usage:**

```bash
python -m scripts.run export path/to/your_best_model.ckpt
```

**Generated Artifacts:**
This command will create an `artifacts/` directory (or the one specified in your config) containing:

- `model.onnx`: The core model graph, optimized for high-performance inference.
- `vocab.json`: A file containing the character and diacritic mappings needed to tokenize text.

These two files are all you need to run the model using the inference package. See the **[Inference Guide](./inference.md)** for how to use them.

---

## 3. The Configuration System

The framework uses a powerful hierarchical configuration system that gives you precise control over every experiment. The system loads and merges settings in a specific order, where later settings override earlier ones.

### 3.1. The Hierarchy of Overrides

Configurations are merged in the following order:

1.  **Base Config (`configs/base.yaml`)**: This file is **always loaded first**. It contains the default settings for the entire pipeline.
2.  **Experiment Config (Optional)**: If you provide an `--experiment path/to/exp.yaml` flag, this file's settings will be merged on top of the base config, overriding any shared keys. This is the recommended way to define specific model architectures and hyperparameters.
3.  **Checkpoint Config (On Resume)**: When resuming with `--ckpt-path`, the config saved inside the checkpoint is used, overriding the base and experiment configs. This ensures a run is resumed with its original settings.
4.  **CLI Arguments (Most Powerful)**: You can override any setting directly from the command line using a `dot.notation` syntax. These arguments are applied last and will override all other configs.

### 3.2. How to Configure a Run

Here are practical examples of how to use the configuration system.

#### Method 1: Using an Experiment File (Recommended)

This is the standard approach. You define your model architecture and learning rates in a dedicated file.

```bash
# The settings in `transformer.yaml` will override the defaults in `base.yaml`
python -m scripts.run train --experiment configs/lstm.yaml
```

#### Method 2: Using Only the Base Config

If you run the `train` command **without** an `--experiment` flag, the system will use only the default settings defined in `configs/base.yaml`.

```bash
# This will run a training session with the default BiLSTM model from base.yaml
python -m scripts.run train
```

#### Method 3: Using CLI Overrides for Quick Experiments

This is perfect for testing small changes without creating a new file. You can override any parameter using the syntax `key.subkey=value`.

```bash
python -m scripts.run train \
  --experiment configs/transformer.yaml \
  data.batch_size=128 \
  trainer.max_epochs=10 \
  modeling_config.optimizer.lr=0.0005
```

In this example:

- `base.yaml` is loaded first.
- `transformer.yaml` overrides it.
- The CLI arguments `data.batch_size=128`, `trainer.max_epochs=10`, and `modeling_config.optimizer.lr=0.0005` are applied last, overriding any values for these keys that were set in the previous files.
