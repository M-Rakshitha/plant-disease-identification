import tensorflow as tf
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from vit_keras.layers import (ClassToken, AddPositionEmbs, TransformerBlock)

# Import ViT custom layers (needed to load ViT-based models)
from vit_keras.layers import (
    ClassToken,
    AddPositionEmbs,
    TransformerBlock,
)

# Tell Keras about custom ViT layers
custom_objects = {
    "ClassToken": ClassToken,
    "AddPositionEmbs": AddPositionEmbs,
    "TransformerBlock": TransformerBlock,
}
custom_objects = {k: v for k, v in custom_objects.items() if v is not None}

# Load your trained model
model = tf.keras.models.load_model(
    "model/model.h5",
    compile=False,
    custom_objects=custom_objects,
    safe_mode=False
)

# Load label names
with open("model/labels.json") as f:
    labels = json.load(f)

# Function to prepare an image for prediction
def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# Ask for image path
image_path = input("Enter the path to a leaf image: ")

# Make prediction
x = prepare_image(image_path)
pred = model.predict(x)[0]
pred_index = int(np.argmax(pred))

# Show result
print(f"\nPrediction: {labels[pred_index]}")
print(f"Confidence: {pred[pred_index]:.3f}\n")

