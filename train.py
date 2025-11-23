import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import build_model

# ---------------- SETTINGS ----------------
DATA_DIR = "data"   # folder with 'train' and 'val'
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS_INITIAL = 15
EPOCHS_FINE = 10
MODEL_DIR = "models"
MODEL_NAME = "efficientnet_b0_brain_tumor.h5"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- DATA PATHS ----------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

# Detect classes
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
NUM_CLASSES = len(classes)
print("Detected classes:", classes)

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' if NUM_CLASSES>2 else 'binary',
    classes=classes,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' if NUM_CLASSES>2 else 'binary',
    classes=classes,
    shuffle=False
)

# ---------------- BUILD MODEL ----------------
model = build_model(num_classes=NUM_CLASSES, input_shape=IMG_SIZE + (3,))
if NUM_CLASSES == 2:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, MODEL_NAME),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# ---------------- INITIAL TRAINING ----------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_INITIAL,
    callbacks=[checkpoint, reduce_lr, early]
)

# ---------------- FINE-TUNE ----------------
base = model.layers[1]  # EfficientNet base
base.trainable = True

for layer in base.layers[:-20]:
    layer.trainable = False

if NUM_CLASSES == 2:
    loss = 'binary_crossentropy'
else:
    loss = 'categorical_crossentropy'

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss,
    metrics=['accuracy']
)

fine_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint, reduce_lr, early]
)

# ---------------- SAVE MODEL ----------------
model.save(os.path.join(MODEL_DIR, MODEL_NAME))
print("✅ Model saved to", os.path.join(MODEL_DIR, MODEL_NAME))
