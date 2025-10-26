import tensorflow as tf
from vit_keras import vit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# 2️⃣ Dataset paths (your dataset should be in "PlantVillage/train" and "PlantVillage/val")
base_dir = "../PlantVillage"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# 3️⃣ Load and preprocess images
IMAGE_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# 4️⃣ Build Vision Transformer model
vit_model = vit.vit_b32(
    image_size=IMAGE_SIZE,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=len(train_data.class_indices)
)

model = tf.keras.Sequential([
    vit_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# 5️⃣ Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 6️⃣ Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=2   # start with 10, increase later
)

# 7️⃣ Save model and labels
model.save("model/model.h5")

with open("model/labels.json", "w") as f:
    json.dump(list(train_data.class_indices.keys()), f)