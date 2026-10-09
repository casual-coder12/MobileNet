import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "imagenette": [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ],
    "oxford_pet": [
        "Abyssinian",
        "American Bulldog",
        "American Pit Bull Terrier",
        "Basset Hound",
        "Beagle",
        "Bengal",
        "Birman",
        "Bombay",
        "Boxer",
        "British Shorthair",
        "Chihuahua",
        "Egyptian Mau",
        "English Cocker Spaniel",
        "English Setter",
        "German Shorthaired",
        "Great Pyrenees",
        "Havanese",
        "Japanese Chin",
        "Keeshond",
        "Leonberger",
        "Maine Coon",
        "Miniature Pinscher",
        "Newfoundland",
        "Persian",
        "Pomeranian",
        "Pug",
        "Ragdoll",
        "Russian Blue",
        "Saint Bernard",
        "Samoyed",
        "Scottish Terrier",
        "Shiba Inu",
        "Siamese",
        "Sphynx",
        "Staffordshire Bull Terrier",
        "Wheaten Terrier",
        "Yorkshire Terrier"
    ]
}


def plot_training_hist_dict(saved_history: dict, dataset_name: str):
    """
    Plots the training and validation loss and accuracy over epochs.
    Handles single-phase and two-phase training results.
    """

    warmup = saved_history["warmup"]
    finetune = saved_history.get("finetune")

    has_finetune = finetune is not None

    if has_finetune:
        loss = warmup["loss"] + finetune["loss"]
        val_loss = warmup["val_loss"] + finetune["val_loss"]
        accuracy = warmup["accuracy"] + finetune["accuracy"]
        val_accuracy = warmup["val_accuracy"] + finetune["val_accuracy"]
        epochs_split = len(warmup["loss"])
    else:
        loss = warmup["loss"]
        val_loss = warmup["val_loss"]
        accuracy = warmup["accuracy"]
        val_accuracy = warmup["val_accuracy"]

    epochs = list(range(1, len(loss) + 1))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training Loss", color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color='red', linewidth=2)

    if has_finetune:
        plt.axvline(
            x=epochs_split + 0.5, 
            color='gray', 
            linestyle='--', 
            linewidth=1.5, 
            label="Start Fine-Tuning"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs ({dataset_name} Dataset)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Training Accuracy", color='green', linewidth=2)
    plt.plot(epochs, val_accuracy, label="Validation Accuracy", color='orange', linewidth=2)

    if has_finetune:
        plt.axvline(
            x=epochs_split + 0.5, 
            color='gray', 
            linestyle='--', 
            linewidth=1.5, 
            label="Start Fine-Tuning"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy over Epochs ({dataset_name} Dataset)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    path_to_save_plots = f"outputs/mobilenet_{dataset_name}_training_plots.png"
    plt.savefig(path_to_save_plots, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(model, dataset, dataset_name = "mnist"):
    """
    Computes and displays a formatted Confusion Matrix heatmap.
    """
    # Extract true labels from test_data and convert predictions to class indices
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    # Make predictions
    y_pred = np.argmax(model.predict(dataset), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    # Define class names based on dataset
    class_names = CLASS_NAMES[dataset_name]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({dataset_name} Dataset)')

    os.makedirs("outputs", exist_ok=True)
    path_to_save_cm = f"outputs/mobilenet_{dataset_name}_confusion_matrix.png"
    plt.savefig(path_to_save_cm, dpi=300, bbox_inches='tight')

    plt.show()
    

def plot_sample_predictions(model, dataset, dataset_name="mnist", num_samples=10):
    """
    Plots a grid of sample images with their true and predicted labels.

    Args:
        model (tf.keras.Model): The trained model used for predictions.
        dataset (tf.data.Dataset): The dataset containing images and labels.
        dataset_name (str): Name of the dataset ("mnist" or "cifar10").
        num_samples (int): Number of samples to display.
    """
    class_names = CLASS_NAMES[dataset_name]

    # Unbatch, shuffle, and take a subset of the dataset for visualization
    sample_ds = dataset.unbatch().shuffle(buffer_size=10000).take(num_samples)
    x_samples, y_true = next(iter(sample_ds.batch(num_samples)))

    # Squeeze to remove single-dimensional entries from the shape
    y_true = np.squeeze(y_true)

    # Make predictions
    y_pred = np.argmax(model.predict(x_samples), axis=1)

    num_samples = min(num_samples, len(x_samples))

    plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        img = x_samples[i]
        img = (img + 1.0) / 2.0
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(img)
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        is_correct = y_true[i] == y_pred[i]
        title_color = "green" if is_correct else "red"
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    path_to_save_samples = f"outputs/mobilenet_{dataset_name}_sample_predictions.png"
    plt.savefig(path_to_save_samples, dpi=300, bbox_inches='tight')

    plt.show()

