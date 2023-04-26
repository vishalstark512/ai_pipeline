import tensorflow as tf
import numpy as np
from PIL import Image
from utils.plot_image import display


class UnetInferrer:
    def __init__(self):
        self.image_size = 128
        self.saved_path = "saved_model/unet.hdf5"
        self.model = tf.keras.saving.load_model(self.saved_path)

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0
    
    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape = tensor_image.shape
        tensor_image = tf.reshape(tensor_image, [1, shape[0], shape[1], shape[2]])
        pred = self.model.predict(tensor_image)
        display([tensor_image[0], pred[0]])
        pred = pred.tolist()
        return {"segmentation_output": pred}
