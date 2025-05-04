import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np

from models.encoder_factory import get_encoder
from models.projection_head import get_projection_head
from utils.augmentations import get_augmentation_pipeline

load_dotenv("config.env")

ENCODER = os.getenv("ENCODER")
PROJECTION_DIM = int(os.getenv("PROJECTION_DIM"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
INPUT_SHAPE = (28, 28, 1)

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1024).batch(BATCH_SIZE).prefetch(1)

base_encoder = get_encoder(ENCODER, INPUT_SHAPE)
projection_head = get_projection_head(base_encoder.output_shape[-1], PROJECTION_DIM)
augment = get_augmentation_pipeline()

optimizer = tf.keras.optimizers.Adam()

# Built-in CosineSimilarity loss function, scaled by temperature
loss_fn = tf.keras.losses.CosineSimilarity(axis=-1)

@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        aug1 = augment(images, training=True)
        aug2 = augment(images, training=True)
        z1 = projection_head(base_encoder(aug1, training=True), training=True)
        z2 = projection_head(base_encoder(aug2, training=True), training=True)
        
        # Compute cosine similarity loss (negative of cosine similarity)
        loss = loss_fn(z1, z2) * -TEMPERATURE  # In contrastive learning, we minimize the negative similarity

    grads = tape.gradient(loss, base_encoder.trainable_variables + projection_head.trainable_variables)
    optimizer.apply_gradients(zip(grads, base_encoder.trainable_variables + projection_head.trainable_variables))
    return loss

for epoch in range(EPOCHS):
    for batch in dataset:
        loss = train_step(batch)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

base_encoder.save("base_encoder.h5")
