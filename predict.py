import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

# -------------------- SETTINGS --------------------
MODEL_PATH = "models/efficientnet_b0_brain_tumor.h5"
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.85  # minimum probability to accept prediction

# Hardcoded labels (keep in sync with your training folders)
LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# -------------------- LOAD MODEL ONCE --------------------
# This ensures fast predictions without reloading the model every time
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# -------------------- PREDICTION FUNCTION --------------------
def predict(img_path):
    """
    Predict the brain tumor type for a given MRI image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        dict: {"label": <predicted_label>, "confidence": <confidence_float>}
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # Predict
    preds = model.predict(x)[0]
    
    # Multi-class classification
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    # Out-of-distribution detection
    if confidence < CONF_THRESHOLD:
        return {"label": "Unrecognized/Invalid Image", "confidence": confidence}

    label = LABELS[idx]
    return {"label": label, "confidence": confidence}

# -------------------- TEST RUN --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print("Prediction:", result)
