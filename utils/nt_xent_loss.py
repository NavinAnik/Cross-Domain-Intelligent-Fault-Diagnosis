
import tensorflow as tf

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)
    z = tf.math.l2_normalize(z, axis=1)

    similarity_matrix = tf.matmul(z, z, transpose_b=True)
    logits = similarity_matrix / temperature

    labels = tf.range(batch_size)
    labels = tf.concat([labels, labels], axis=0)

    mask = tf.eye(2 * batch_size)
    logits_mask = 1 - mask
    logits = logits * logits_mask

    positives = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0)
    labels = positives

    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)
