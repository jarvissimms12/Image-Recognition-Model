import os
import shutil
import cv2
import random
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# --- Configuration ---
# Ensure these folders exist in your repository or are downloaded
input_folder = 'Agricultural-crops' 
output_folder = 'Processed_Data'
img_size = (224, 224)
batch_size = 32

# --- Data Preparation ---
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

splitfolders.ratio(input_folder, output=output_folder, seed=500, ratio=(0.8, 0.1, 0.1))

# --- Data Generator ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Validation and Test only need preprocessing ---
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_datagen.flow_from_directory(
    os.path.join(output_folder, 'train'),
    target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

valid_data = val_test_datagen.flow_from_directory(
    os.path.join(output_folder, 'val'),
    target_size=img_size, batch_size=batch_size, class_mode='categorical'

# --- Model Building (Transfer Learning) ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(30, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Training ---
history = model.fit(train_data, epochs=10) # Set higher for final training

# --- Saving ---
model.save('CropModel.keras')
print("Model saved successfully!")
