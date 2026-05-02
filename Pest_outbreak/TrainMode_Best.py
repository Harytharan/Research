import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import constants as constants

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
NON_PEST_LABELS = {"none", "not_pest", "non_pest"}


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)


def collect_dataset_classes(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    class_counts = {}
    for entry in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(class_path):
            continue

        image_count = sum(
            1
            for filename in os.listdir(class_path)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )
        class_counts[entry] = image_count

    if not class_counts:
        raise ValueError(f"No class folders found in dataset: {dataset_path}")

    non_empty_classes = [name for name, count in class_counts.items() if count > 0]
    empty_classes = [name for name, count in class_counts.items() if count == 0]

    if empty_classes:
        print(f"[WARN] Skipping empty class folders: {', '.join(empty_classes)}")

    if not non_empty_classes:
        raise ValueError("No images found in any class folder.")

    return class_counts, non_empty_classes, empty_classes


def build_generators(dataset_path, class_names):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
        validation_split=constants.VALIDATION_SPLIT,
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=constants.VALIDATION_SPLIT,
    )

    generator_args = {
        "directory": dataset_path,
        "target_size": constants.IMG_SIZE,
        "batch_size": constants.BATCH_SIZE,
        "class_mode": "categorical",
        "classes": class_names,
    }

    train_data = train_datagen.flow_from_directory(
        subset="training",
        shuffle=True,
        **generator_args,
    )

    validation_data = validation_datagen.flow_from_directory(
        subset="validation",
        shuffle=False,
        **generator_args,
    )

    return train_data, validation_data


def build_model(num_classes):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=constants.INITIAL_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(class_counts, class_names):
    total_samples = sum(class_counts[name] for name in class_names)
    class_count = len(class_names)
    weights = {}

    for index, class_name in enumerate(class_names):
        sample_count = class_counts[class_name]
        weights[index] = total_samples / (class_count * sample_count)

    return weights


def unfreeze_top_layers(model, unfreeze_layers):
    head_start_index = next(
        (
            index
            for index, layer in enumerate(model.layers)
            if isinstance(layer, GlobalAveragePooling2D)
        ),
        None,
    )

    if head_start_index is None:
        raise ValueError("Could not locate the custom classification head for fine-tuning.")

    backbone_layers = model.layers[:head_start_index]
    head_layers = model.layers[head_start_index:]

    for layer in backbone_layers:
        layer.trainable = False

    for layer in backbone_layers[-unfreeze_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    for layer in head_layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=constants.FINE_TUNE_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def create_callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def build_checkpoint_path(model_path):
    model_dir = os.path.dirname(model_path)
    model_name, model_ext = os.path.splitext(os.path.basename(model_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(model_dir, f"{model_name}.checkpoint_{timestamp}{model_ext}")


def finalize_model_file(source_model_path, target_model_path):
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Checkpoint model not found: {source_model_path}")

    try:
        if os.path.exists(target_model_path):
            os.replace(source_model_path, target_model_path)
        else:
            os.replace(source_model_path, target_model_path)
        print(f"Best model saved to {target_model_path}")
        return target_model_path
    except OSError as exc:
        print(
            "[WARN] Could not replace the existing model file. "
            f"It is likely open in another app. Newly trained model kept at: {source_model_path}"
        )
        print(f"[WARN] Replace failed for {target_model_path}: {exc}")
        return source_model_path


def main():
    dataset_path = resolve_path(constants.DATASET_PATH)
    model_path = resolve_path(constants.MODEL_SAVE_PATH)
    class_indices_path = resolve_path(constants.CLASS_INDICES_PATH)
    checkpoint_path = build_checkpoint_path(model_path)

    class_counts, class_names, empty_classes = collect_dataset_classes(dataset_path)

    print("[INFO] Training on the following pest classes:")
    for class_name in class_names:
        print(f"  - {class_name}: {class_counts[class_name]} images")

    if any(label.lower() in NON_PEST_LABELS for label in empty_classes):
        print("[WARN] The non-pest class folder exists but has no images, so it will not be learned.")

    train_data, validation_data = build_generators(dataset_path, class_names)
    model = build_model(train_data.num_classes)
    class_weights = compute_class_weights(class_counts, class_names)

    print(f"[INFO] Training classifier head for {constants.HEAD_EPOCHS} epochs...")
    head_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=constants.HEAD_EPOCHS,
        callbacks=create_callbacks(checkpoint_path),
        class_weight=class_weights,
    )

    print(f"[INFO] Fine-tuning top {constants.UNFREEZE_LAYERS} ResNet layers for {constants.FINE_TUNE_EPOCHS} epochs...")
    model = unfreeze_top_layers(model, constants.UNFREEZE_LAYERS)
    fine_tune_history = model.fit(
        train_data,
        validation_data=validation_data,
        initial_epoch=constants.HEAD_EPOCHS,
        epochs=constants.HEAD_EPOCHS + constants.FINE_TUNE_EPOCHS,
        callbacks=create_callbacks(checkpoint_path),
        class_weight=class_weights,
    )

    history = merge_histories(head_history, fine_tune_history)

    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)

    loss, accuracy = model.evaluate(validation_data)
    print(f"\nValidation Accuracy: {accuracy:.2f}")

    with open(class_indices_path, "w", encoding="utf-8") as file_handle:
        json.dump(train_data.class_indices, file_handle, indent=2)
    print(f"Saved class indices to {class_indices_path}")

    saved_model_path = checkpoint_path
    if not os.path.exists(saved_model_path):
        model.save(saved_model_path)

    finalize_model_file(saved_model_path, model_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs. Validation Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
