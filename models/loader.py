import os

import pandas as pd
import tensorflow as tf

def load_model(model_class, dataset_name, load_type="b", input_shape=(32, 32, 3), num_classes=10, saved_models_dir="saved_models"):
    """
    Loads a model either by instantiating `model_class` and loading weights (.weights.h5), 
    or by loading the entire model architecture + weights (.keras).
    """
    ext = ".weights.h5" if load_type == "w" else ".keras"
    file_path = os.path.join(saved_models_dir, f"mobilenet_{dataset_name}{ext}")
    file_path_best = os.path.join(saved_models_dir, f"mobilenet_{dataset_name}_best_model.keras")

    if load_type == "w":
        model = model_class(input_shape=input_shape, num_classes=num_classes)
        model.load_weights(file_path)
        print(f"Model weights successfully loaded from {file_path}")
    elif load_type == "m":
        model = tf.keras.models.load_model(file_path)
        print(f"Model successfully loaded from {file_path}")
    elif load_type == "b":
        model = tf.keras.models.load_model(file_path_best)
        print(f"Best model successfully loaded from {file_path_best}")
    else:
        raise ValueError("Invalid load_type. Choose 'w', 'm', or 'b'.")

    return model


def load_history_from_csv(history_path: str, warmup_epochs: int = 5) -> dict:
    """
    Loads training history from a CSV file and splits it into warmup and fine-tuning phases.
    Returns a dictionary with keys 'warmup' and 'finetune', each containing the corresponding training metrics.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"File not found at path: {history_path}")

    df = pd.read_csv(history_path)

    # Check if the DataFrame has enough rows to split into warmup and fine-tuning phases
    if len(df) > warmup_epochs:
        df_warmup = df.iloc[:warmup_epochs]
        df_finetune = df.iloc[warmup_epochs:]
        return {
            "warmup": df_warmup.to_dict(orient="list"),
            "finetune": df_finetune.to_dict(orient="list")
        }
    else:
        # If the DataFrame has fewer rows than warmup_epochs, treat all as warmup and return None for finetune
        return {
            "warmup": df.to_dict(orient="list"),
            "finetune": None
        }