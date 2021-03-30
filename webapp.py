import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)



# Loading the images
#@st.cache
#def load_image(image_file):
#    img = Image.open(image_file)
#    return img
@st.cache(allow_output_mutation=True)
def load_model():
    seg_model = 'unet_with_densenet_weights-23-0.4608.hdf5'
    loaded_model.load_weights(seg_model)
    loaded_model.summary()  # included to make it visible when model is reloaded
    #session = K.get_session()
    return loaded_model
 

header = st.beta_container()
with header:
    st.title("Welcome to my Project - Pneumothorax Prediction App")

    st.markdown(unsafe_allow_html=True, body="<p>Welcome to Pneumothorax Prediction APP.</p>"
                                             "<p>This is a basic app built with Streamlit."
                                             "With this app, you can upload a Chest X-Ray image and predict if the patient "
                                             "from that image suffers pneumonia or not.</p>"
                                             "<p>The model used is a Convolutional Neural Network (CNN)")


# select the model
st.sidebar.subheader("Input")
models_list = ["seg_model"]
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
        uploaded_file  = st.file_uploader("Upload a Image", type=['dcm', 'jpeg', 'png', 'jpg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Pneumothorax image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            # st.image(load_image(uploaded_file))
            # st.text("You haven't uploaded image file")
            # To See Details
            st.write(type(uploaded_file))
            # st.write(dir(image_file))
            file_details = {"Filename": image_file.name,
                            "FileType": image_file.type,
                            "FileSize": image_file.size}
            st.write(file_details)


            st.write("")
            model = load_model()

            if st.button('predict'):
                rslt_1 = model.predict(image_pred.reshape(1,224,224,3))
                rslt = rslt_1.argmax(axis=1)[0]
                label = "Please consult with your doctor , The patient has Pneumothorax" if rslt == 0 else "Not Pneumothorax"
                st.warning(label)
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
