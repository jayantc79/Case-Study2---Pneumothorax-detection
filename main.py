import streamlit as st
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image 



st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "unet_with_densenet_weights-23-0.4608.hdf5"
  model_loader = load_model(fp)
  return model_loader# Loading the images

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

seg_model = loading_model()

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
        image_file = st.file_uploader("Upload a dicom Image", type=['dcm', 'jpeg', 'png', 'jpg'])
        if image_file is not None:
            st.text("You haven't uploaded image file")
            # To See Details
            st.write(type(image_file))
            # st.write(dir(image_file))
            file_details = {"Filename": image_file.name,
                            "FileType": image_file.type,
                            "FileSize": image_file.size}
            st.write(file_details)

            st.image(load_image(image_file))
            img = image.load_img(image_file.name, target_size=(500, 500),color_mode='grayscale')
            image1 = image.img_to_array(img)
            image1 = image1/255
            image1 = np.expand_dims(image1, axis=0)

            st.write("")

            #predict
            preds= seg_model.predict(image1)
            if preds>= 0.5:
              out = ('I am {:.2%} percent confirmed that the patient doesnt have Pneumothorax'.format(preds[0][0]))
            else:
              out = ('I am {:.2%} percent confirmed that the patient has Pneumothorax'.format(1-preds[0][0])
                
            st.success(out)              
            image = Image.open(image_file)
            st.image(image,use_column_width=True)

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
