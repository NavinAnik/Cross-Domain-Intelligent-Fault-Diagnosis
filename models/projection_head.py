import tensorflow as tf

def get_projection_head(input_dim, projection_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(projection_dim)(x)
    return tf.keras.Model(inputs, outputs, name="projection_head")
