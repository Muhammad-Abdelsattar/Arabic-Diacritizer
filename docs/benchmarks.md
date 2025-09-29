# Benchmarks & Performance

This document provides performance metrics for the pre-trained models available in this repository and compares them against other existing solutions.

Our goal is to provide models that strike the best balance between **high accuracy** and **computational efficiency**, making them suitable for real-world production and edge deployment.

## Dataset used for training

The training dataset is this corpus [Training Data](https://drive.google.com/file/d/1shhIEKc2FITVorSX26cmN9dPkKkYxI48/view?usp=sharing) from the [Hareef](https://github.com/mush42/hareef) repo.
Plus some additional data from the [Arabic News Dataset](https://huggingface.co/datasets/arbml/Arabic_News) that was **synthetically diacritized** using **Gemini-2.5-flash** which made the dataset well balanced between Classical and Modern Standard Arabic.
The synthetically diacritized dataset is available for download from the [Arabic Diacritization Synthetic Dataset](https://huggingface.co/datasets/Muhammad7777/Synthetic-Arabic-Diacritization-Dataset).

## Evaluation Metrics & Dataset

All models are evaluated on the standard **SadeedDiac-25** benchmark (unless otherwise noted) and compared to the performance LLMs evaluated on this benchmark.
The benchmark is a comprehensive and linguistically diverse benchmark specifically designed for evaluating Arabic diacritization models. It unifies Modern Standard Arabic (MSA) and Classical Arabic (CA) in a single dataset, addressing key limitations in existing benchmarks.
The benchmark is available for download from the [Sadeed Benchmark](https://huggingface.co/datasets/Misraj/SadeedDiac-25) repository.

Also, the models are evaluated on the test set of the [Arabic Diacritizer Evaluation](https://github.com/AliOsm/arabic-text-diacritization) repo and compared to multiple existing systems of the smae scale.

- **DER (Diacritic Error Rate):** The percentage of characters that were assigned the wrong diacritic. **Lower is better.**
- **Size:** The disk size of the exported ONNX model.

---

## 1. Pre-trained Models

These are the official models available for automatic download via the `inference` package. They are optimized for ONNX Runtime.

| Model Size | Architecture    | DER (%) (Sadeed Benchmark) | ONNX Size (MB)         | Speed (CPU) |
| :--------- | :-------------- | :------------------------- | :--------------------- | :---------- |
| **Small**  | BiLSTM w/ hints | 3.05%                      | 4 MB (1 M params)      | **Fastest** |
| **Medium** | BiLSTM w/ Hints | 2.54%                      | 15.5 MB (3.9 M params) | Fast        |

> Note: These values are based on the SadeedDiac-25 benchmark.

> Note: All the current and future models would be available via [Pre-trained Models](https://huggingface.co/Muhammad7777/arabic-diacritizer-models).

> Note: Models were trained with diacritics hints, however, they were evaluated without hints.

---

## 2. Comparison with models From AliOsm/arabic-text-diacritization experiments

Here is how our models compare to models from the `AliOsm/arabic-text-diacritization` repository.

The `AliOsm/arabic-text-diacritization` repository is a collection of models trained for Arabic text diacritization. It provides evaluation for existing system on a provided test data on the same repo. Their benchmark and data can be found [Here](https://github.com/AliOsm/arabic-text-diacritization).

Here is how our models compare to models from the `AliOsm/arabic-text-diacritization` benchmark.

| System                 | DER (%)   |
| :--------------------- | :-------- |
| **This Repo (Medium)** | **1.23%** |
| **This Repo (small)**  | **1.52%** |
| shakkala               | 4.36%     |
| mishkal                | 17.59%    |
| farasa                 | 24.9%     |
| harakat                | 20.64%    |

---

## 3. Comparison with LLMs From SadeedDiac-25 benchmark

Here is how our models compare to LLMs from the SadeedDiac-25 benchmark.

| System                   | DER (%) |
| :----------------------- | :------ |
| **This Repo (Medium)**   | 2.54%   |
| **This Repo (small)**    | 3.05%   |
| Claude-3-7-Sonnet-Latest | ~1.39%  |
| GPT-4                    | ~3.86%  |
| Gemini-Flash-2.0         | ~3.19%  |

---

## Notes

**1. DER (%) is the diacritization error rate.**

**2. All the evaluations were done with excluding the no-diacritic label.**

**3. The evaluations of the models at the beginning were done on the test set of the SadeedDiac-25 benchmark.**
