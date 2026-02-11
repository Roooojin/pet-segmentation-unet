import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 128
BATCH_SIZE = 16
SEED = 42

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_unet.keras")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 3

# ---------- Metrics (must match training) ----------
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

# ---------- Dataset (test only) ----------
(ds, info) = tfds.load("oxford_iiit_pet", with_info=True)
test_raw = ds["test"]

def to_image_mask(example):
    return example["image"], example["segmentation_mask"]

def preprocess(image, mask):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    mask  = tf.image.resize(mask,  (IMG_SIZE, IMG_SIZE), method="nearest")
    image = tf.cast(image, tf.float32) / 255.0
    mask  = tf.cast(mask, tf.int32) - 1
    mask  = tf.squeeze(mask, axis=-1)
    return image, mask

test_ds = (test_raw
           .map(to_image_mask, num_parallel_calls=AUTOTUNE)
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# ---------- Load model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nFirst run: python train.py")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mean_iou": mean_iou, "dice_coef": dice_coef}
)
print("Loaded model:", MODEL_PATH)

# ---------- Evaluate ----------
# results = model.evaluate(test_ds, verbose=0)
# print("\nEvaluation on test set:")
# for name, val in zip(model.metrics_names, results):
#     print(f"{name}: {val:.4f}")
# (Optional but recommended) re-compile to force metrics to be tracked explicitly
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[mean_iou, dice_coef],
)

results = model.evaluate(test_ds, verbose=0, return_dict=True)

print("\nEvaluation on test set:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")


# ---------- Save prediction images ----------
def save_predictions(ds, n=8):
    saved = 0
    for images, masks in ds.take(1):
        preds = model.predict(images, verbose=0)
        pred_mask = tf.argmax(preds, axis=-1)

        n = min(n, images.shape[0])
        for i in range(n):
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(images[i])
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("GT Mask")
            plt.imshow(masks[i], cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Pred Mask")
            plt.imshow(pred_mask[i], cmap="gray")
            plt.axis("off")

            out_path = os.path.join(OUTPUT_DIR, f"pred_{i}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print("Saved:", out_path)
            saved += 1

    print(f"\nSaved {saved} prediction images in: {OUTPUT_DIR}")

save_predictions(test_ds, n=8)
