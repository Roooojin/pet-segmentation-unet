import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# Config
# ======================
IMG_SIZE = 128
NUM_CLASSES = 3
MODEL_PATH = os.path.join("outputs", "best_unet.keras")

st.set_page_config(page_title="Pet Segmentation (U-Net)", layout="wide")
st.title("🐶🐱 Pet Segmentation with U-Net (Oxford-IIIT Pet)")
st.write("Upload an image → model predicts segmentation mask → shows overlay (no re-training).")

# ======================
# Metrics (needed for load_model)
# ======================
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true_oh = tf.one_hot(y_true, depth=NUM_CLASSES, dtype=tf.float32)
    y_pred_oh = tf.one_hot(y_pred, depth=NUM_CLASSES, dtype=tf.float32)

    axes = [0, 1, 2]
    inter = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)
    union = tf.reduce_sum(y_true_oh + y_pred_oh, axis=axes) - inter

    iou = (inter + 1e-7) / (union + 1e-7)
    return tf.reduce_mean(iou)

def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true_oh = tf.one_hot(y_true, depth=NUM_CLASSES, dtype=tf.float32)
    y_pred_oh = tf.one_hot(y_pred, depth=NUM_CLASSES, dtype=tf.float32)

    axes = [0, 1, 2]
    inter = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)
    denom = tf.reduce_sum(y_true_oh + y_pred_oh, axis=axes)

    dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
    return tf.reduce_mean(dice)

# ======================
# Load model once (cache)
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"First run training once to create outputs/best_unet.keras"
        )
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"mean_iou": mean_iou, "dice_coef": dice_coef}
    )
    return model

model = load_model()
st.success(f"Loaded model: {MODEL_PATH}")

# ======================
# Upload UI
# ======================
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

colA, colB = st.columns([1, 1])

with colA:
    alpha = st.slider("Overlay transparency", 0.0, 1.0, 0.45, 0.05)
    show_border_only = st.checkbox("Show border class only (class=2)", value=False)

# ======================
# Predict
# ======================
def preprocess_pil(pil_img: Image.Image):
    pil_rgb = pil_img.convert("RGB")
    pil_resized = pil_rgb.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(pil_resized).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # [1,H,W,3]
    return pil_rgb, pil_resized, x

def predict_mask(x):
    pred = model.predict(x, verbose=0)           # [1,H,W,C]
    mask = np.argmax(pred[0], axis=-1).astype(np.uint8)  # [H,W]
    return mask

def render_results(orig, resized, mask, alpha=0.45, border_only=False):
    # choose what to visualize
    vis_mask = mask.copy()
    if border_only:
        vis_mask = (vis_mask == 2).astype(np.uint8)  # 0/1 mask

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(vis_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(resized)
    plt.imshow(vis_mask, alpha=alpha)
    plt.axis("off")

    plt.tight_layout()
    return fig

if uploaded is not None:
    pil_img = Image.open(uploaded)
    orig, resized, x = preprocess_pil(pil_img)
    mask = predict_mask(x)

    # basic info
    with colB:
        st.write("Mask value meanings (trained on Oxford-IIIT Pet):")
        st.write("- 0: background")
        st.write("- 1: pet")
        st.write("- 2: border")

    fig = render_results(orig, resized, mask, alpha=alpha, border_only=show_border_only)
    st.pyplot(fig)

    # Save output option
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "ui_result.png")
    fig.savefig(out_path, dpi=150)
    st.info(f"Saved latest result to: {out_path}")
else:
    st.warning("Upload an image to see prediction.")
