import tensorflow as tf

def build_classifier(base_encoder, num_classes):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = base_encoder(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)
