import tensorflow as tf
from tensorflow.keras import layers, models

def MobileNet(input_shape, num_classes, augmentation=False):
    """
    Builds the MobileNet architecture.
    """

    # Define the input
    X_input = layers.Input(input_shape)

    if augmentation:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
            layers.RandomTranslation(0.05, 0.05)
            ])
        X_input = data_augmentation(X_input)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
        )
    base_model.trainable = False  # Freeze the base model

    X = base_model(X_input, training=False)

    X = layers.GlobalAveragePooling2D()(X)
    X = layers.Dropout(0.3)(X)  # Dropout layer to reduce overfitting
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create model
    model = models.Model(inputs=X_input, outputs=X, name='MobileNet')

    return model

