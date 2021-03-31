import cv2
from PIL import Image
import os
import pandas as pd
import tensorflow as tf
import streamlit as st
from tensorflow import keras
import pydicom
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy



seg_model = "unet_with_densenet_weights-23-0.4608.hdf5"
IMG_WIDTH = IMG_HEIGHT = 256
N_CHANNELS=3

@st.cache(allow_output_mutation=True)
def load_model():
    seg_model = load_model('unet_with_densenet_weights-23-0.4608.hdf5')
    # add anything else you want to do to the model here
    seg_model.load_weights('unet_with_densenet_weights-23-0.4608.hdf5')
    return seg_model
#def load_model():
  #print("loading model")
  #model = tf.keras.models.load_model("seg_model", compile=True)

  #return model
def predict(X,Y):
  img = tf.io.read_file(X)
  image = tf.image.decode_png(img)
  image = tf.image.convert_image_dtype(image, tf.float32)#converting the image to tf.float32
  image=tf.squeeze(image,[0]) #squeezing the image because the file is of the shape(1,1024,1024,1)and we want (1024,1024,3)
  b = tf.constant([1,1,3], tf.int32)
  image=tf.tile(image,b)#the image is of the shape (1024,1024,1) to make it (1024,1024,3) and for this using tf.tile
  image=tf.image.resize(image,size=[256,256])
  image=tf.expand_dims(image,axis=0)
  if Y!=" -1":
    print("Ground truth of Classification is 1(Has Pneumothorax)")
    print('*'*100)
  else:
    print("Ground truth of Classification is 0(Does not have Pneumothorax)")
    print("Ground truth of Segmentation -- There is no mask")
    print('*'*100)

    
  if model.predict(image)>=0.5:
    print("Pneumothorax has been detected")
    mask=final.predict(image)
    mask=(mask>0.5).astype(np.uint8)
    try:
      true_mask=Image.fromarray(mask_functions.rle2mask(Y,1024,1024).T).resize((256,256), resample=Image.BILINEAR)
      true_mask=np.array(true_mask)
      plt.figure(figsize=(20,6))
      plt.subplot(121)
      plt.title("X-ray image with mask(Ground truth)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(true_mask),cmap='gray',alpha=0.3)
      plt.subplot(122)
      plt.title("X-ray image with mask(Predicted)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(mask),cmap='gray',alpha=0.3)
      return plt.show()
    except: #if there is no ground truth mask
      plt.figure(figsize=(20,6))
      plt.title("X-ray image with mask(Predicted)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(mask),cmap='gray',alpha=0.3)
      return plt.show()
