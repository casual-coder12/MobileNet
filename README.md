# MobileNetV2 Image Classification

A TensorFlow/Keras image-classification project that applies transfer learning with an ImageNet-pretrained MobileNetV2 backbone. The project includes dataset pipelines, a two-phase training workflow, model checkpoints, evaluation utilities, and a notebook with results for three datasets.

## Overview

The classifier replaces MobileNetV2's original classification head with global average pooling, dropout, and a dataset-specific softmax layer. Training first learns the new head while the pretrained backbone is frozen, then fine-tunes the backbone at a lower learning rate.

## Features

- **Transfer learning with MobileNetV2**: Starts from ImageNet-pretrained weights.
- **Three datasets**: CIFAR-10, Imagenette, and Oxford-IIIT Pet.
- **Two-phase training**: Frozen-backbone warmup followed by fine-tuning.
- **Data augmentation**: Optional random horizontal flips, rotations, and translations.
- **Evaluation metrics**: Sparse categorical accuracy and top-2 accuracy, with loss.
- **Training utilities**: Validation, early stopping, learning-rate reduction, best-validation-loss checkpoints, and CSV history.
- **Visualizations**: Training history plots, confusion matrices, and sample predictions.

## Model Architecture

The model uses `tf.keras.applications.MobileNetV2` with `include_top=False` and ImageNet weights. Its original classifier is replaced with:

1. Global average pooling
2. Dropout (`0.3`)
3. A dataset-specific dense softmax classification layer

The training script enables augmentation by default. The trainer freezes the MobileNetV2 backbone for the warmup phase, then unfreezes it for fine-tuning with a smaller learning rate.

## Datasets

| Dataset | Classes | Project split and preprocessing |
|---|---:|---|
| CIFAR-10 | 10 | Uses the dataset's test split; reserves 10% of the training split for validation; resizes images to 224×224. |
| Imagenette | 10 | Uses the validation split for testing; reserves 10% of the training split for validation; resizes images to 224×224. |
| Oxford-IIIT Pet | 37 | Uses the dataset's test split; reserves 10% of the training split for validation; resizes images to 224×224. |

All three pipelines apply MobileNetV2 preprocessing, scaling image values to the range expected by the pretrained backbone.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Train

Train on Imagenette (the default dataset):

```bash
python train.py
```

Choose a dataset explicitly:

```bash
python train.py --dataset imagenette
python train.py --dataset oxford_pet
```

Training options include `--w_epochs`, `--f_epochs`, `--batch_size`, `--head_lr`, `--fine_tune_lr`, `--save_type` (`w`, `m`, or `both`), and `--load_checkpoint`. Run `python train.py --help` for the available options and defaults.

**CIFAR-10 input-size note:** the dataset loader resizes images to 224×224, while `train.py` currently constructs the CIFAR-10 model with a 32×32 input shape. These sizes need to be made consistent before using the training CLI with CIFAR-10.

### Evaluate

Evaluate a saved best Imagenette model:

```bash
python evaluate.py --dataset imagenette --load_type b
```

The evaluator supports loading weights (`w`), a complete saved model (`m`), or the best-validation-loss model (`b`). The current command-line choices are CIFAR-10 and Imagenette. Oxford-IIIT Pet results are available in the notebook; it is not currently an accepted `evaluate.py --dataset` choice.

## Results

The following test-set metrics are reported by `notebooks/MobileNet.ipynb`:

| Dataset | Test accuracy | Test loss |
|---|---:|---:|
| Imagenette | 97.83% | 0.0763 |
| CIFAR-10 | 85.58% | 0.4269 |
| Oxford-IIIT Pet | 87.46% | 0.3806 |

These are the results recorded in the notebook and may differ from new training or evaluation runs.

## Project Structure

```text
.
├── data/
│   ├── dataset.py              # Dataset loading and preprocessing pipelines
│   └── dataset_copy.py         # Alternate dataset pipeline implementation
├── models/
│   ├── model.py                # MobileNetV2 transfer-learning model
│   └── loader.py               # Model and training-history loading
├── notebooks/
│   ├── MobileNet.ipynb         # Evaluation notebook and recorded results
│   └── outputs/                # Notebook visualizations
├── outputs/                    # Generated evaluation visualizations
├── saved_models/               # Best models and CSV training histories
├── utils/
│   ├── trainer.py              # Training, fine-tuning, saving, and evaluation
│   └── visualize.py            # Training and evaluation visualizations
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
└── requirements.txt            # Python dependencies
```

## Generated Files

Training writes best-validation-loss model checkpoints and CSV history files to `saved_models/`. Depending on `--save_type`, it can also save model weights, the complete model, or both. Training and evaluation visualizations are written to `outputs/`.

## Dependencies

The main dependencies are TensorFlow/Keras, NumPy, Matplotlib, Seaborn, scikit-learn, Hugging Face `datasets(v2.21.0)`. See `requirements.txt` for the complete list.

---

*Neural Networks and Deep Learning — MobileNetV2 Project*
