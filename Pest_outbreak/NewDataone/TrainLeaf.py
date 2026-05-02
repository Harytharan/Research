import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "leafDataset")
MODEL_PATH = os.path.join(BASE_DIR, "paddy_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
NON_LEAF_LABELS = {"none", "not_leaf", "non_leaf"}
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 16
EPOCHS = 20


def validate_dataset(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    class_names = sorted(
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    )

    if not class_names:
        raise ValueError(f"No class folders found in dataset: {dataset_dir}")

    normalized_names = {name.lower() for name in class_names}
    if not normalized_names.intersection(NON_LEAF_LABELS):
        raise ValueError(
            "Add a non-leaf class folder named one of: none, not_leaf, non_leaf"
        )

    print("Detected classes:", ", ".join(class_names))


validate_dataset(DATASET_DIR)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model and class indices
model.save(MODEL_PATH)

with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved class indices to: {CLASS_INDICES_PATH}")