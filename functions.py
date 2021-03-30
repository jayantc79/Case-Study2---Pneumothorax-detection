import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, InceptionV3, Xception, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Activation, Add, multiply, add, concatenate, LeakyReLU, ZeroPadding2D, UpSampling2D, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
seg_model = "unet_with_densenet_weights-23-0.4608.hdf5"
IMG_WIDTH = IMG_HEIGHT = 256
N_CHANNELS=3

@st.cache(allow_output_mutation=True)
def load_model():
  print("loading model")
  model = tf.keras.models.load_model("seg_model", compile=True)

  return model


def preprocess_image(file):
  img = tf.io.read_file(file)
  img = tf.image.decode_png(img, channels= N_CHANNELS)
  img = tf.image.convert_image_dtype(img, tf.float32) 
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 
  img.set_shape((IMG_HEIGHT,IMG_WIDTH,3))


def predict(model, file):
  prob = model.predict(np.reshape(file, [1, 224, 224, 3]))

  if prob > 0.5:
    prediction = True
  else:
    prediction = False

  return prob, prediction
