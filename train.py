# import os
# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
#
# # =============================
# # Config
# # =============================
# AUTOTUNE = tf.data.AUTOTUNE
# IMG_SIZE = 128
# BATCH_SIZE = 16
# EPOCHS = 12
# SEED = 42
#
# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# MODEL_PATH = os.path.join(OUTPUT_DIR, "best_unet.keras")
#
# tf.random.set_seed(SEED)
# np.random.seed(SEED)
#
# print("TensorFlow:", tf.__version__)
# print("TFDS:", tfds.__version__)
# print("Saving best model to:", MODEL_PATH)
#
# # =============================
# # Dataset
# # =============================
# (ds, info) = tfds.load("oxford_iiit_pet", with_info=True)
# train_raw = ds["train"]
# test_raw = ds["test"]
#
# NUM_CLASSES = 3  # masks 1..3 -> shift to 0..2
#
# def to_image_mask(example):
#     return example["image"], example["segmentation_mask"]
#
# train_raw = train_raw.map(to_image_mask, num_parallel_calls=AUTOTUNE)
# test_raw  = test_raw.map(to_image_mask, num_parallel_calls=AUTOTUNE)
#
# def preprocess(image, mask):
#     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
#     mask  = tf.image.resize(mask,  (IMG_SIZE, IMG_SIZE), method="nearest")
#     image = tf.cast(image, tf.float32) / 255.0
#     mask  = tf.cast(mask, tf.int32) - 1  # keep [H,W,1]
#     return image, mask
#
# def augment(image, mask):
#     if tf.random.uniform(()) > 0.5:
#         image = tf.image.flip_left_right(image)
#         mask  = tf.image.flip_left_right(mask)
#     image = tf.image.random_brightness(image, 0.12)
#     image = tf.image.random_contrast(image, 0.85, 1.15)
#     return image, mask
#
# def finalize_mask(image, mask):
#     mask = tf.squeeze(mask, axis=-1)  # [H,W]
#     return image, mask
#
# train_ds = (train_raw
#             .map(preprocess, num_parallel_calls=AUTOTUNE)
#             .map(augment, num_parallel_calls=AUTOTUNE)
#             .map(finalize_mask, num_parallel_calls=AUTOTUNE)
#             .shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
#             .batch(BATCH_SIZE)
#             .prefetch(AUTOTUNE))
#
# test_ds = (test_raw
#            .map(preprocess, num_parallel_calls=AUTOTUNE)
#            .map(finalize_mask, num_parallel_calls=AUTOTUNE)
#            .batch(BATCH_SIZE)
#            .prefetch(AUTOTUNE))
#
# # sanity check
# for images, masks in train_ds.take(1):
#     print("Sanity check:", images.shape, masks.shape, np.unique(masks[0].numpy()))
#
# # =============================
# # U-Net model
# # =============================
# def conv_block(x, filters):
#     x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     return x
#
# def unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     inputs = tf.keras.Input(shape=input_shape)
#
#     c1 = conv_block(inputs, 32);  p1 = tf.keras.layers.MaxPool2D()(c1)
#     c2 = conv_block(p1, 64);      p2 = tf.keras.layers.MaxPool2D()(c2)
#     c3 = conv_block(p2, 128);     p3 = tf.keras.layers.MaxPool2D()(c3)
#     c4 = conv_block(p3, 256);     p4 = tf.keras.layers.MaxPool2D()(c4)
#
#     bn = conv_block(p4, 512)
#
#     u4 = tf.keras.layers.UpSampling2D()(bn)
#     u4 = tf.keras.layers.Concatenate()([u4, c4])
#     c5 = conv_block(u4, 256)
#
#     u3 = tf.keras.layers.UpSampling2D()(c5)
#     u3 = tf.keras.layers.Concatenate()([u3, c3])
#     c6 = conv_block(u3, 128)
#
#     u2 = tf.keras.layers.UpSampling2D()(c6)
#     u2 = tf.keras.layers.Concatenate()([u2, c2])
#     c7 = conv_block(u2, 64)
#
#     u1 = tf.keras.layers.UpSampling2D()(c7)
#     u1 = tf.keras.layers.Concatenate()([u1, c1])
#     c8 = conv_block(u1, 32)
#
#     outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(c8)
#     return tf.keras.Model(inputs, outputs)
#
# # =============================
# # Metrics (vectorized, TF2.20-safe)
# # =============================
# def mean_iou(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.int32)
#     y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
#
#     y_true_oh = tf.one_hot(y_true, depth=NUM_CLASSES, dtype=tf.float32)
#     y_pred_oh = tf.one_hot(y_pred, depth=NUM_CLASSES, dtype=tf.float32)
#
#     axes = [0, 1, 2]
#     inter = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)
#     union = tf.reduce_sum(y_true_oh + y_pred_oh, axis=axes) - inter
#
#     iou = (inter + 1e-7) / (union + 1e-7)
#     return tf.reduce_mean(iou)
#
# def dice_coef(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.int32)
#     y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
#
#     y_true_oh = tf.one_hot(y_true, depth=NUM_CLASSES, dtype=tf.float32)
#     y_pred_oh = tf.one_hot(y_pred, depth=NUM_CLASSES, dtype=tf.float32)
#
#     axes = [0, 1, 2]
#     inter = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)
#     denom = tf.reduce_sum(y_true_oh + y_pred_oh, axis=axes)
#
#     dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
#     return tf.reduce_mean(dice)
#
# # =============================
# # Train
# # =============================
# model = unet()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[mean_iou, dice_coef]
# )
#
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
#     tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=MODEL_PATH,
#         monitor="val_mean_iou",
#         mode="max",
#         save_best_only=True
#     )
# ]
#
# history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
#
# # =============================
# # Save plots (no plt.show to avoid blocking)
# # =============================
# def save_plot(y1, y2, title, ylabel, filename):
#     plt.figure(figsize=(7, 4))
#     plt.plot(y1, label="train")
#     plt.plot(y2, label="val")
#     plt.title(title)
#     plt.xlabel("epoch")
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, filename)
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print("Saved:", path)
#
# save_plot(history.history["loss"], history.history["val_loss"], "Loss", "loss", "loss.png")
# save_plot(history.history["mean_iou"], history.history["val_mean_iou"], "Mean IoU", "IoU", "iou.png")
# save_plot(history.history["dice_coef"], history.history["val_dice_coef"], "Dice", "Dice", "dice.png")
#
# print("\nTraining finished.")
# print("Best model saved at:", MODEL_PATH)


import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# =============================
# Config
# =============================
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_unet.keras")  # توجه: این مدل قبلی رو overwrite می‌کنه

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TensorFlow:", tf.__version__)
print("TFDS:", tfds.__version__)
print("Saving best model to:", MODEL_PATH)

# =============================
# Dataset (Train / Val / Test)
# =============================
NUM_CLASSES = 3  # masks 1..3 -> shift to 0..2
VAL_PCT = 10     # 10% of train for validation

(splits, info) = tfds.load(
    "oxford_iiit_pet",
    split=[f"train[:{100-VAL_PCT}%]", f"train[{100-VAL_PCT}%:]", "test"],
    with_info=True,
    shuffle_files=False,  # deterministic split
)

train_raw, val_raw, test_raw = splits

def to_image_mask(example):
    return example["image"], example["segmentation_mask"]

train_raw = train_raw.map(to_image_mask, num_parallel_calls=AUTOTUNE)
val_raw   = val_raw.map(to_image_mask, num_parallel_calls=AUTOTUNE)
test_raw  = test_raw.map(to_image_mask, num_parallel_calls=AUTOTUNE)

def preprocess(image, mask):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    mask  = tf.image.resize(mask,  (IMG_SIZE, IMG_SIZE), method="nearest")
    image = tf.cast(image, tf.float32) / 255.0
    mask  = tf.cast(mask, tf.int32) - 1  # keep [H,W,1] -> values 0..2
    return image, mask

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, 0.12)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return image, mask

def finalize_mask(image, mask):
    mask = tf.squeeze(mask, axis=-1)  # [H,W]
    return image, mask

train_ds = (train_raw
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .map(finalize_mask, num_parallel_calls=AUTOTUNE)
            .shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (val_raw
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .map(finalize_mask, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

test_ds = (test_raw
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .map(finalize_mask, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# sanity check
for images, masks in train_ds.take(1):
    print("Sanity check (train):", images.shape, masks.shape, np.unique(masks[0].numpy()))
for images, masks in val_ds.take(1):
    print("Sanity check (val):  ", images.shape, masks.shape, np.unique(masks[0].numpy()))

# =============================
# U-Net model
# =============================
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32);  p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 64);      p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 128);     p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 256);     p4 = tf.keras.layers.MaxPool2D()(c4)

    bn = conv_block(p4, 512)

    u4 = tf.keras.layers.UpSampling2D()(bn)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(c8)
    return tf.keras.Model(inputs, outputs)

# =============================
# Metrics
# =============================
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

# =============================
# Train
# =============================
model = unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[mean_iou, dice_coef]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_mean_iou",
        mode="max",
        save_best_only=True
    )
]

# ✅ مهم: validation_data الان val_ds است (نه test_ds)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# =============================
# Save plots
# =============================
def save_plot(y1, y2, title, ylabel, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(y1, label="train")
    plt.plot(y2, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

save_plot(history.history["loss"], history.history["val_loss"], "Loss", "loss", "loss.png")
save_plot(history.history["mean_iou"], history.history["val_mean_iou"], "Mean IoU", "IoU", "iou.png")
save_plot(history.history["dice_coef"], history.history["val_dice_coef"], "Dice", "Dice", "dice.png")

print("\nTraining finished.")
print("Best model saved at:", MODEL_PATH)
