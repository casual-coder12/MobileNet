import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models.model import MobileNet

class MobileNetTransferTrainer:
    """
    Trainer class optimized for Transfer Learning and Fine-Tuning 
    a pretrained MobileNet architecture in TensorFlow/Keras.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        base_model_name: str = "MobileNetV2",
        head_lr: float = 1e-3,
        fine_tune_lr: float = 2e-5,
        loss_fn=None,
    ):
        """
        Initializes the trainer with model, optimizer, and loss function.

        Args:
            model (tf.keras.Model): The MobileNet model instance to be trained.
            base_model_name (str): Name of the base MobileNet model for transfer learning.
            head_lr (float): Learning rate for the head layers. Default is 1e-3.
            fine_tune_lr (float): Learning rate for fine-tuning. Default is 2e-5.
            fine_tune_lr (float): Learning rate for fine-tuning. Default is 2e-5.
            loss_fn: Keras loss function instance. Defaults to CategoricalCrossentropy if None.
        """
        self.model = model
        self.base_model_name = base_model_name
        self.head_lr = head_lr
        self.fine_tune_lr = fine_tune_lr

        # Set default loss function to CategoricalCrossentropy (assuming one-hot encoded labels)
        self.loss_fn = loss_fn or losses.SparseCategoricalCrossentropy()

        try:
            self.base_model = self.model.get_layer(self.base_model_name)
        except ValueError:
            # If the base model layer is not found by name, attempt to find it by type
            self.base_model = next(
                (l for l in self.model.layers if isinstance(l, tf.keras.Model)), 
                None
                )

        self.metric_fns=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            ]

    def _compile_model(self, learning_rate):
        """
        Compiles the model with the specified learning rate.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss_fn,
            metrics=self.metric_fns,
        )

    def freeze_base_model(self):
        """
        Freezes the base model layers to prevent them from being updated during training.
        """
        if self.base_model is not None:
            self.base_model.trainable = False
            print(f"Base model '{self.base_model_name}' frozen for transfer learning.")
        else:
            # If the base model is not found, freeze all layers except the last few layers (assuming they are the classifier)
            for layer in self.model.layers[:-2]:  # Freeze all but the last 5 layers
                layer.trainable = False
            print("Base model not found by name. Freezing all but the last 2 layers for transfer learning.")

        self._compile_model(self.head_lr)

    def unfreeze_base_model(self, fine_tune_at: int = None):
        """
        Unfreezes the base model layers to allow fine-tuning.
        """
        if self.base_model is not None:
            self.base_model.trainable = True
            print(f"Base model '{self.base_model_name}' unfrozen for fine-tuning.")
        else:
            # If the base model is not found, unfreeze layers from a specific point
            if fine_tune_at is not None:
                for layer in self.model.layers[fine_tune_at:]:
                    layer.trainable = True
                print(f"Layers from index {fine_tune_at} unfrozen for fine-tuning.")
            else:
                # Unfreeze all layers if no specific point is provided
                for layer in self.model.layers:
                    layer.trainable = True
                print("All layers unfrozen for fine-tuning.")

        # Compile the model with a lower learning rate for fine-tuning
        self._compile_model(self.fine_tune_lr)

    def train_transfer(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        dataset_name: str,
        save_type: str = "both",
        warmup_epochs: int = 5,
        fine_tune_epochs: int = 10,
        fine_tune_at: int = None,
    ) -> dict:
        """
        Trains the MobileNet model using transfer learning and fine-tuning.

        Args:
            train_data (tf.data.Dataset): Prepared training dataset loader.
            val_data (tf.data.Dataset): Prepared validation dataset loader.
            dataset_name (str): Name of the dataset for saving the model.
            warmup_epochs (int): Number of epochs to train with frozen base model. Default is 5.
            fine_tune_epochs (int): Number of epochs to fine-tune the entire model. Default is 10.
            fine_tune_at (int): Layer index from which to start fine-tuning. If None, all layers are unfrozen.
        Returns:
            dict: Training history containing loss and accuracy metrics.
        """

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")

        path_history = os.path.join("saved_models", f"mobilenet_{dataset_name}_history.csv")
        path_checkpoint = os.path.join("saved_models", f"mobilenet_{dataset_name}_best_model.keras")

        callbacks = [
            CSVLogger(path_history, append=True),
            ModelCheckpoint(path_checkpoint, monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        ]

        # Step 1: Train with frozen base model
        print(f"--- Step 1: Training Model with Frozen Base for {warmup_epochs} Epochs ---")
        self.freeze_base_model()
        history_warmup = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=warmup_epochs,
            callbacks=callbacks
        )

        history_fine_tune = None

        # Step 2: Fine-tune the entire model
        if fine_tune_epochs > 0:
            print(f"--- Step 2: Fine-Tuning Model for {fine_tune_epochs} Epochs ---")
            self.unfreeze_base_model(fine_tune_at=fine_tune_at)
            
            total_epochs = warmup_epochs + fine_tune_epochs
            history_fine_tune = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=total_epochs,
                initial_epoch=warmup_epochs,
                callbacks=callbacks
            )
            
        self.save_model(dataset_name=dataset_name, save_type=save_type)

        return {"warmup": history_warmup.history, "finetune": history_fine_tune.history}

    def evaluate(self, test_data: tf.data.Dataset, dataset_name: str) -> dict:
        """
        Quick Evaluation of the trained model on a test dataset.

        Returns:
            dict: Evaluation metrics containing test loss and accuracy scores.
        """
        print(f"--- Evaluating MobileNet on ({dataset_name}) Test Data ---")
        results = self.model.evaluate(test_data, return_dict=True)

        return results

    def predict(self, dataset: tf.data.Dataset, max_batches: int=None):
        """
        Extracts images, true labels, and predicted class probabilities/indices in a SINGLE pass.
        
        Returns:
            images (np.ndarray): Flattened array of input images.
            y_true (np.ndarray): 1D array of true class indices.
            y_pred (np.ndarray): 1D array of predicted class indices.
            raw_predictions (np.ndarray): 2D array of raw prediction probabilities.
        """
        x_list, y_list = [], []

        # Iterating through batches in a SINGLE pass
        for i, (x_batch, y_batch) in enumerate(dataset):
            if max_batches is not None and i >= max_batches:
                break
            x_list.append(x_batch.numpy())
            y_list.append(y_batch.numpy())

        x_samples = np.concatenate(x_list, axis=0)
        y_true_raw = np.concatenate(y_list, axis=0)

        # Making predictions on collected images
        raw_predictions = self.model.predict(x_samples, verbose=0)

        # Converting labels and predictions to 1D indices (0-9)
        y_true = np.argmax(y_true_raw, axis=1) if y_true_raw.ndim > 1 else y_true_raw
        y_pred = np.argmax(raw_predictions, axis=1)

        return x_samples, y_true, y_pred, raw_predictions

    def save_model(self, dataset_name: str, save_type: str = 'both'):
        """
        Saves the model either as weights or the entire model.

        Args:
            dataset_name (str): Name of the dataset for saving the model.
            save_type (str): Type of save operation: 'w' or 'm' or 'both'
        """

        path_weights = os.path.join("saved_models", f"mobilenet_{dataset_name}.weights.h5")
        path_model = os.path.join("saved_models", f"mobilenet_{dataset_name}.keras")

        if save_type == "w":
            self.model.save_weights(path_weights)
            print(f"\n=====Model Weights Successfully Saved at {path_weights}=====")
        elif save_type == "m":
            self.model.save(path_model)
            print(f"\n=====Entire Model Successfully Saved at {path_model}=====")
        elif save_type == "both":
            self.model.save_weights(path_weights)
            self.model.save(path_model)
            print(f"\n=====Model Weights Successfully Saved at {path_weights}=====")
            print(f"\n=====Entire Model Successfully Saved at {path_model}=====")
        else:
            raise ValueError("Invalid save_type. Choose 'w', 'm' or 'both'.")

    @staticmethod
    def load_model(model_class, dataset_name, load_type="b", input_shape=(32, 32, 3), num_classes=10, saved_models_dir="saved_models"):
        """
        Loads a MobileNet model or its weights from the specified directory.
        """
        ext = ".weights.h5" if load_type == "w" else ".keras"
        file_path = os.path.join(saved_models_dir, f"mobilenet_{dataset_name}{ext}")
        file_path_best = os.path.join(saved_models_dir, f"mobilenet_{dataset_name}_best_model.keras")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

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
    
