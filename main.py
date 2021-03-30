import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "unet_with_densenet_weights-23-0.4608.hdf5"
  model_loader = load_model(fp)
  return model_loader

seg_model = loading_model()
st.write("""
# Pneumothorax Prediction 
by Jayant C. :)
""")



  


temp = st.file_uploader("Upload X-Ray Image")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:

 

  img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  image1 = image.img_to_array(img)
  image1 = image1/255
  image1 = np.expand_dims(image1, axis=0)

  #predict
  preds= seg_model.predict(image1)
  if preds>= 0.5:
    out = ('I am {:.2%} percent confirmed that Pneumothorax has been detected'.format(preds[0][0]))
  
  else: 
    out = ('I am {:.2%} percent confirmed that No Pneumothorax Detected'.format(1-preds[0][0]))

  st.success(out)
  
  image1 = Image.open(temp)
  st.image(image1,use_column_width=True)
