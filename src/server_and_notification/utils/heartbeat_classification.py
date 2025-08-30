# pylint: skip-file

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_labels = [
    "ATRIAL PREMATURE BEAT",
    "LEFT BUNDLE BRANCH BEAT",
    "NORMAL",
    "PACED BEAT",
    "PREMATURE VENTRICULAR CONTRACTION",
    "RIGHT BUNDLE BRANCH BEAT",
    "VENTRICULAR ESCAPE BEAT",
    "VENTRICULAR FLUTTER WAVE"
]

def classify_heartbeat(image_path):
    model = load_model('../cnn/models/arrythmia_detection_cnn.keras')

    img = image.load_img(
        image_path,
        target_size=(128, 128),
        color_mode='grayscale'
    )

    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class
    probability_predictions = model.predict(image_array)  # get probabilities for each class

    output = np.argmax(probability_predictions[0])  # index of highest probability = 0
    output_name = class_labels[output]  # use class_labels map for full name

    if output_name != 'NORMAL':
        print(
            "\n"
            "***************************************************\n"
            "*                                                 *\n"
            "*  ⚠️  WARNING: ABNORMAL HEARTBEAT DETECTED  ⚠️  *\n"
            "*                                                 *\n"
           f"*       TYPE: {output_name}                       *\n"
            "*                                                 *\n"
            "***************************************************\n"
        )
    else:
        print("HEARTBEAT NORMAL - ALL CLEAR!")
