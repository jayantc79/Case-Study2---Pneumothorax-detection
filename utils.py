import tensorflow as tf

def read_image(path):
    """
    This reads image 
    ----------------
    path : image path
    ----------------
    """

    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)
    # x = x / 255.0
    # x = tf.image.convert_image_dtype(x, tf.float32)

    return x

def read_mask(path):
    """
    This reads mask 
    ----------------
    path : mask path
    ----------------
    """
    y = tf.io.read_file(path)
    y = tf.image.decode_png(y, channels=1)
    y = tf.image.convert_image_dtype(y, tf.float32)
    # y = y / 255.0
    # y = tf.image.convert_image_dtype(y, tf.float32)
    return y
