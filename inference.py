from utils import read_image
from model import build_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class inference():
    """This is pipeline for predicting the Pneumothorax"""
    def __init__(self, weights_path, input_shape=(3,256,256)):
        # intialize the model and load saved weights
        self.model = build_model(input_shape)
        self.model.load_weights(weights_path)

    def Predict(self, image_path, plot):
        """
        This function predicts the mask for given input image and plots it.
        -------------------------------------------------------------------
        image_path : image path 
        plot       : Boolean (True or False)
        -------------------------------------------------------------------
        if plot is True then plots the image with predicted mask if False 
        returns predicted mask array
        """
        # read the original image
        image_orig = read_image(image_path)
        # reshape image and mask as first channel image format
        image = tf.transpose(image_orig, [2,0,1])
        # predict the mask using trained model
        predict_mask = self.model.predict(tf.expand_dims(image, axis=0))
        predict_mask = tf.transpose(predict_mask, [0,2,3,1])
        if plot == True:
            plt.imshow(image_orig)      
            plt.imshow(np.squeeze(predict_mask[:,:,:,1]), cmap='Reds', alpha = 0.3)
        # if plot == False return the predicted mask 
        else:
            return predict_mask
