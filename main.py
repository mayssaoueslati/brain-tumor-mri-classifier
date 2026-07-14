# %%
import kagglehub
import os

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
train_path = os.path.join(path, "Training")
test_path = os.path.join(path, "Testing")

print("Train classes:")
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    print(folder, len(os.listdir(folder_path)))

print("\nTest classes:")
for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    print(folder, len(os.listdir(folder_path)))

# %%
import matplotlib.pyplot as plt
from PIL import Image
import random

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, cls in enumerate(os.listdir(train_path)):
    cls_folder = os.path.join(train_path, cls)
    samples = random.sample(os.listdir(cls_folder), 5)
    for j, fname in enumerate(samples):
        img = Image.open(os.path.join(cls_folder, fname))
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].set_title(f"{cls} {img.size} {img.mode}")
        axes[i, j].axis("off")
plt.tight_layout()
plt.show()

# %%
from collections import Counter

sizes = Counter()
modes = Counter()
for cls in os.listdir(train_path):
    cls_folder = os.path.join(train_path, cls)
    for fname in os.listdir(cls_folder)[:200]:
        img = Image.open(os.path.join(cls_folder, fname))
        sizes[img.size] += 1
        modes[img.mode] += 1

print("Sizes:", sizes.most_common(10))
print("Modes:", modes)

# %%
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training set (85%) + Validation set (15%) — split from the Training folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=123,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=123,
    shuffle=True
)

# Test set — completely separate folder, never touched during training
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())

# %%
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# %%
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')   # 4 classes, softmax for multi-class probability output
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',   # matches label_mode="categorical" from your data pipeline
    metrics=['accuracy']
)

model.summary()
# %%
# %%
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)
# %%
model.save("brain_tumor_cnn_v1.keras")
print("Model saved.")
# %%
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
# %%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report: precision, recall, F1 per class
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# %%
