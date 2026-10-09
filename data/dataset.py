import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, utils
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def prepare_cifar10_dataset(batch_size=64, buffer_size=10000, image_size=(224, 224)):
    """
    Loads the CIFAR-10 dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    dataset = load_dataset("uoft-cs/cifar10")

    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)

    X_train = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in train_val["train"]["img"]])
    y_train = np.asarray(train_val["train"]["label"])
    
    X_val = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in train_val["test"]["img"]])
    y_val = np.asarray(train_val["test"]["label"])

    X_test = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in dataset["test"]["img"]])
    y_test = np.asarray(dataset["test"]["label"])

    print("CIFAR-10 Raw Shapes:", X_train.shape, X_val.shape, X_test.shape, )  # Debugging shapes

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=buffer_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

    def preprocess_fn(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, image_size)  # Resizovanje samo za tekući batch
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # MobileNet [-1, 1]
        return image, label
    
    train_dataset = train_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = test_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def prepare_imagenette_dataset(batch_size=64, buffer_size=10000, image_size=(224, 224)):
    """
    Loads the Imagenette dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    dataset = load_dataset("frgfm/imagenette", "160px", trust_remote_code=True)
    
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)

    X_train = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["train"]["image"]])
    y_train = np.asarray(train_val["train"]["label"], dtype=np.int32)

    X_val = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["test"]["image"]])
    y_val = np.asarray(train_val["test"]["label"], dtype=np.int32)

    X_test = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in dataset["validation"]["image"]])
    y_test = np.asarray(dataset["validation"]["label"], dtype=np.int32)

    print("Imagenette Raw Shapes:", X_train.shape, X_val.shape, X_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=min(buffer_size, len(X_train)))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def preprocess_fn(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # MobileNet [-1, 1]
        return image, label

    train_dataset = train_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def prepare_oxford_pet_dataset(batch_size=64, buffer_size=10000, image_size=(224, 224)):
    """
    Loads the Oxford-IIIT Pet dataset and prepares it for training and testing.

    Args:
        batch_size (int): Number of samples per batch.
        image_size (tuple): The size to resize images to.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.

    """
    dataset = load_dataset("timm/oxford_iiit_pet")
    
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)

    X_train = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["train"]["image"]])
    y_train = np.asarray(train_val["train"]["label"])

    X_val = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in train_val["test"]["image"]])
    y_val = np.asarray(train_val["test"]["label"])

    X_test = np.stack([np.asarray(img.convert("RGB").resize(image_size), dtype=np.uint8) for img in dataset["test"]["image"]])
    y_test = np.asarray(dataset["test"]["label"])

    print("Oxford Pet Dataset Raw Shapes:", X_train.shape, X_val.shape, X_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=min(buffer_size, len(X_train)))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def preprocess_fn(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # MobileNet [-1, 1]
        return image, label

    train_dataset = train_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size).map(
        preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

