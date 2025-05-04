import tensorflow as tf

def get_encoder(name, input_shape):
    if name == "basic_cnn":
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return tf.keras.Model(inputs, x, name="basic_cnn_encoder")
    else:
        raise ValueError(f"Encoder '{name}' not recognized.")
