# pylint: skip-file

import tensorflow as tf

def create_cnn():
    """
    Builds and returns a CNN model.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.5),  # Prevents overfitting
        tf.keras.layers.Dense(8, activation='softmax')  # Output layer (num of classes, activation function)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
