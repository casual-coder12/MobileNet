import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses, metrics, models

from data.dataset import prepare_cifar10_dataset, prepare_imagenette_dataset, prepare_oxford_pet_dataset
from models.model import MobileNet
from models.loader import load_history_from_csv, load_model
from utils.trainer import MobileNetTransferTrainer
from utils.visualize import plot_confusion_matrix, plot_sample_predictions, plot_training_hist_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained MobileNet model on MNIST or CIFAR-10 datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagenette"],
        help="Dataset name to evaluate on"
    )
    parser.add_argument(
        "--load_type",
        type=str,
        default="b",
        choices=["w", "m", "b"],
        help="Type of load operation: 'w' for weights, 'm' for entire model or 'b' for best model (default: b).",
        )
    return parser.parse_args()

def evaluate():
    """
    Evaluate the model on the test dataset and print the results.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_data (tf.data.Dataset): The test dataset.
    """
    args = parse_args()
    dataset_name = args.dataset

    outputs_dir = "outputs"
    saved_models_dir = "saved_models"
    os.makedirs(outputs_dir, exist_ok=True)
    
    if dataset_name == "cifar10":
        _, _, test_data = prepare_cifar10_dataset(batch_size=64)
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset_name == "imagenette":
            _, _, test_data = prepare_imagenette_dataset(batch_size=64)
            input_shape = (160, 160, 3)
            num_classes = 10
    elif dataset_name == "oxford_pet":
            _, _, test_data = prepare_oxford_pet_dataset(batch_size=64)
            input_shape = (224, 224, 3)
            num_classes = 37
    else:
        raise ValueError("Invalid dataset name. Choose 'cifar10'.")

    # Load the trained model
    model = load_model(
        model_class=MobileNet,
        dataset_name=dataset_name,
        load_type=args.load_type,
        input_shape=input_shape,
        num_classes=num_classes,
        saved_models_dir=saved_models_dir
    )

    trainer = MobileNetTransferTrainer(model=model)

    # Evaluate the model on the test dataset
    results = trainer.evaluate(test_data, dataset_name)

    print("\n" + "="*50)
    print(" " * 15 + "Evaluation Results" + " " * 15)
    print("=" * 50 + "\n")
    print(f"Test Loss: {results}")
    print(f"Test Loss: {results["loss"]:.4f}")
    print(f"Test Accuracy: {results["accuracy"]*100:.2f}%")

    history_path = os.path.join(saved_models_dir, f"mobilenet_{dataset_name}_history.csv")

    history_dict = load_history_from_csv(history_path, warmup_epochs=5)
    plot_training_hist_dict(history_dict, dataset_name=dataset_name)

    plot_confusion_matrix(model, test_data, dataset_name=dataset_name)

    plot_sample_predictions(trainer, test_data, dataset_name=dataset_name)

if __name__ == "__main__":
    evaluate()