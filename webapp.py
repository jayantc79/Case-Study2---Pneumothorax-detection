import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


IMG_WIDTH = IMG_HEIGHT = 256
N_CHANNELS=3

st.set_option('deprecation.showfileUploaderEncoding', False)


model_weights = "densenet121_weights-33-0.9072.hdf5"

def load_weights():
    print("loading model")
    dense_model = load_weights('model_weights')
    return dense_model

# Loading the images
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def final_pred1(image):
  img = tf.io.read_file(image)
  image = tf.image.decode_png(img, channels = N_CHANNELS)
  image = tf.image.convert_image_dtype(image, tf.float32)#converting the image to tf.float32
  image=tf.image.resize(image,size=[256,256])
  image=tf.expand_dims(image,axis=0)
   #recall
  if dense_model.predict(image)>=0.5:
    print("Pneumothorax has been detected")
    mask=final.predict(image)
    mask=(mask>0.5).astype(np.uint8)
    plt.figure(figsize=(20,6))
    plt.title("Mask")
    return plt.imshow(np.squeeze(mask),cmap='gray')
  else:
    return "No Pneumothorax Detected"

header = st.beta_container()
with header:
    st.title("Welcome to my Project - Pneumothorax Prediction App")

    st.markdown(unsafe_allow_html=True, body="<p>Welcome to Pneumothorax Prediction APP.</p>"
                                             "<p>This is a application built with Streamlit."
                                             "With this app, you can upload a Chest X-Ray image and predict if the patient "
                                             "from that image suffers pneumothorax or not.</p>"
                                             "<p>The model used is a Convolutional Neural Network (CNN)")


# select the model
st.sidebar.subheader("Input")
models_list = ["dense_model"]
network = st.sidebar.selectbox("Select the Model", models_list)

def main():
    html_temp = """ 
            <div style="background-color:tomato;padding:10px"> 
            <h2 style="color:white;text-align:center;">Streamlit Pneumothorax Prediction App </h2> 
            </div> 
            """
    st.markdown(html_temp, unsafe_allow_html=True)
    # st.header("Pneumothorax dataset")
    menu = ["Home", "Dataset"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload a dicom Image", type=['dcm', 'jpeg', 'png', 'jpg'])
        if image_file is not None:
            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            file_details = {"Filename": image_file.name,
                            "FileType": image_file.type,
                            "FileSize": image_file.size}
            st.write(file_details)
            st.text("Uploaded Image")

            st.image(load_image(image_file))

            st.write("")

            if st.button('predict'):
                state.text('Predicting...') 
                pred_mask= final_pred1.predict(tf.expand_dims(image_file,axis=0)).reshape((IMG_HEIGHT,IMG_WIDTH)) 
                pred_mask = cv2.resize(pred_mask,(1024,1024))
                pred_mask = (pred_mask > .5).astype(int)
                img = cv2.resize(img.numpy(),(1024,1024))
                col1, col2,col3 = st.beta_columns(3)
                col1.header("Original")
                col1.image(img, use_column_width=True)
                col2.header("Prediction")
                col2.image(pred_mask*255, use_column_width=True)
                state.text('Prediction Done')
                st.write("Result...")

    elif choice == "Dataset":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
                st.write(file_details)

                df = pd.read_csv(data_file)
                st.dataframe(df)

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("This is pneumothorax app")
        st.text("Jayant C.")

if __name__ == '__main__':
    main()
