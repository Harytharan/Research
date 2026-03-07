import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import constants as constants
# Define constants


# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# No augmentation for testing, just rescaling
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training and test datasets
train_data = train_datagen.flow_from_directory(
    constants.TRAIN_PATH,
    target_size=constants.IMG_SIZE,
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    constants.TEST_PATH,
    target_size=constants.IMG_SIZE,
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical'
)

# Load Pretrained ResNet50 (exclude top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)  # Pooling layer
x = Dense(256, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout should be applied AFTER Dense
output = Dense(train_data.num_classes, activation='softmax')(x)  # Final classification layer

# Define the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(constants.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=constants.EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Save final model
model.save("final_pest_model.h5")
print("Model saved as final_pest_model.h5")

# Plot Training vs. Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs. Validation Accuracy')
plt.show()
