import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, input_shape=(224,224,3), dropout=0.3):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )
    base.trainable = False  # freeze for initial training

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
