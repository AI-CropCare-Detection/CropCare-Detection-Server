import tensorflow as tf
import numpy as np


def processing_new_image(path_to_image):
    image = tf.keras.preprocessing.image.load_img(path_to_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    return input_arr