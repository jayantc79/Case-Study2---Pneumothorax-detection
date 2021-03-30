import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models load_model



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
