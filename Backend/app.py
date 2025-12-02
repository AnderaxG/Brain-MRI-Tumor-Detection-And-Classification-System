# app.py
import os
import io
import logging
from pathlib import Path
from typing import Dict

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.applications import VGG16

# ---- Config ----
PROJECT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = PROJECT_DIR / "models"
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 128  # must match training IMAGE_SIZE
NUM_CLASSES = 4

# IMPORTANT: keep the label order same as training
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

BEST_H5 = MODEL_DIR / "best_model.h5"
FINAL_KERAS = MODEL_DIR / "final_model.keras"
WEIGHTS_H5 = MODEL_DIR / "model_weights.h5"  # optional

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit

# ---- Flask app ----
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_EXTENSIONS"] = ALLOWED_EXT

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mri-flask-app")


def build_classification_model(image_size: int = IMAGE_SIZE, num_classes: int = NUM_CLASSES) -> Model:
    """
    Rebuilds the VGG16-based model architecture that was used for training:
    VGG16 (imagenet, include_top=False) -> Flatten -> Dropout -> Dense(128) -> Dropout -> Dense(num_classes, softmax)
    """
    base = VGG16(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    # Freeze base by default - inference only
    base.trainable = False

    x = base.output
    x = Flatten(name="flatten_custom")(x)
    x = Dropout(0.5, name="dropout_custom")(x)
    x = Dense(128, activation="relu", name="dense_custom")(x)
    x = Dropout(0.5, name="dropout_custom_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name="mri_vgg16_classifier")
    return model


def try_load_model() -> Model:
    """
    Attempts to load the model in this order:
     1) load_model(BEST_H5)
     2) load_model(FINAL_KERAS)
     3) rebuild architecture and load weights by_name from best_model.h5
    Raises RuntimeError if not successful.
    """
    logger.info("Model directory: %s", MODEL_DIR)
    # 1) try H5 full model (best_model.h5)
    if BEST_H5.exists():
        try:
            logger.info("Trying to load H5 model: %s", BEST_H5)
            model = load_model(str(BEST_H5), compile=False)
            logger.info("Loaded H5 model from %s", BEST_H5)
            return model
        except Exception as e:
            logger.warning("Failed loading H5 full model: %s", e)

    # 2) try native .keras saved model
    if FINAL_KERAS.exists():
        try:
            logger.info("Trying to load Keras native file: %s", FINAL_KERAS)
            model = load_model(str(FINAL_KERAS), compile=False)
            logger.info("Loaded native Keras model from %s", FINAL_KERAS)
            return model
        except Exception as e:
            logger.warning("Failed loading final_model.keras: %s", e)

    # 3) rebuild architecture & load weights by_name from best_model.h5 (works in many cases)
    if BEST_H5.exists() or WEIGHTS_H5.exists():
        try:
            logger.info("Rebuilding architecture and attempting to load weights by_name.")
            model = build_classification_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
            source = BEST_H5 if BEST_H5.exists() else WEIGHTS_H5
            logger.info("Loading weights (by_name=True) from: %s", source)
            model.load_weights(str(source), by_name=True)
            logger.info("Weights loaded into rebuilt architecture.")
            return model
        except Exception as e:
            logger.warning("Failed to load weights into rebuilt architecture: %s", e)

    raise RuntimeError("Could not load a model from the models/ directory. Please ensure best_model.h5 or final_model.keras exists.")


# Load model at startup (fail fast)
try:
    MODEL = try_load_model()
    logger.info("Model loaded and ready for inference.")
except Exception as e:
    logger.exception("Model load failed at startup: %s", e)
    # keep MODEL as None - predict route will return error if called
    MODEL = None


# ---- Utilities ----
def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXT


def preprocess_image(file_stream: io.BytesIO, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Read file_stream (binary) -> load image -> resize -> to array -> scale [0,1]
    Returns numpy array shaped (1, H, W, 3)
    """
    img = load_img(file_stream, target_size=(image_size, image_size))
    arr = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ---- Routes ----
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": f"File extension not allowed. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    try:
        # read into BytesIO so we can pass to Keras loader
        file_stream = io.BytesIO(file.read())
        img_arr = preprocess_image(file_stream, image_size=IMAGE_SIZE)
        preds = MODEL.predict(img_arr, verbose=0)[0]  # shape: (num_classes,)
        # ensure length matches class names
        if len(preds) != len(CLASS_NAMES):
            logger.warning("Prediction length (%d) != CLASS_NAMES (%d)", len(preds), len(CLASS_NAMES))

        probs_by_name: Dict[str, float] = {name: float(preds[idx]) for idx, name in enumerate(CLASS_NAMES)}
        top_idx = int(np.argmax(preds))
        top_label = CLASS_NAMES[top_idx]
        top_prob = float(preds[top_idx])

        response = {
            "all_probs": probs_by_name,
            "predicted_index": top_idx,
            "predicted_label": top_label,
            "probability": top_prob,
        }
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Prediction failed on server.", "details": str(e)}), 500


# ---- Run ----
if __name__ == "__main__":
    # Use debug=False for normal use; set to True while developing.
    app.run(host="0.0.0.0", port=5000, debug=False)