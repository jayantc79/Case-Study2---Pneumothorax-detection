import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#################################################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("<h1 style='text-align: center; color: black; font-size : 3rem'>Pneumothorax Prediction App</h2>", unsafe_allow_html=True)
st.markdown(unsafe_allow_html=True, body="<br><p>Welcome to Pneumothorax Prediction App.</p>"
                                         "<p>This is a basic app built with Streamlit."
                                         "With this app, you can upload a Chest X-Ray image and predict if the patient "
                                         "suffers from pneumonia or not.</p>"
                                         "<p>The model used is a Convolutional Neural Network (CNN).")
html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Pneumothorax Prediction App </h2>
        </div><br>
        """
st.markdown(html_temp, unsafe_allow_html=True)

##################################################################################################

@st.cache
def load_model():
    model = tf.keras.models.load_model('densenet121.hdf5')
    return model


def final_fun(X, model):
    #img = tf.io.read_file(X)
    image = tfio.image.decode_dicom_image(X, dtype=tf.uint8,color_dim=True,scale='preserve')
    image = tf.image.convert_image_dtype(image, tf.float32)#converting the image to tf.float32
    image=tf.squeeze(image,[0]) #squeezing the image because the file is of the shape(1,1024,1024,1) and we want (1024,1024,3)
    b = tf.constant([1,1,3], tf.int32)
    image=tf.tile(image,b)#the image is of the shape (1024,1024,1) to make it (1024,1024,3) I am using tf.tile
    image=tf.image.resize(image,size=[256,256])
    image=tf.expand_dims(image,axis=0)
    #recall
    if model.predict(image)>=0.5:
        st.error("Pneumothorax has been detected!")

    else:
        st.success("No Pneumothorax Detected!")


def main():
    menu = ['Home','Dataset']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        st.subheader("Home")

        image_file = st.file_uploader(label="Upload Image", type=['dcm'])

        if image_file is not None:
            file_details = {
                                "Filename": image_file.name,
                                "FileType": image_file.type,
                                "FileSize": image_file.size
                           }

            st.write(file_details)

            image = pydicom.dcmread(image_file)
            plt.figure(figsize=(5,5))
            plt.imshow(image.pixel_array, cmap=plt.cm.bone)
            plt.axis('off')
            st.pyplot()

        if st.button('Predict'):
            if image_file is None:
                st.error('No image uploaded :(')

            else:
                model = load_model()
                final_fun(image_file.getvalue(), model)


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
                st.error('No file uploaded :(')

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("This is pneumothorax app")
        st.text("Jayant C.")


if __name__ == '__main__':
    main()
