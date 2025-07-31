import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import Counter



base_path = os.path.dirname(__file__)  # directory of the script
csv_path = os.path.join(base_path, "feedback", "feedback_log.csv")

df = pd.read_csv(csv_path, names=["filepath", "predicted", "correct"])
images, labels = [], []

# Load and preprocess images
for i, row in df.iterrows():
    img = Image.open(row["filepath"]).convert("RGB").resize((224, 224))
    images.append(np.array(img))
    labels.append(row["correct"])

# Encode data
X = np.array(images) / 255.0
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

# Check: are there at least 2 samples per class and 2+ classes?
label_counts = Counter(y)

if num_classes < 2 or any(count < 2 for count in label_counts.values()):
    print("❌ Not enough distinct labels with multiple samples to train a model.")
    print("👉 Please submit at least 2 samples for each of 2 or more labels.")
    exit()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Build model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=output)

# Freeze base layers
for layer in base.layers:
    layer.trainable = False

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Save model and label encoder
model.save("custom_clothing_model.h5")
np.save("class_labels.npy", encoder.classes_)

print("✅ Combined model trained and saved.")
