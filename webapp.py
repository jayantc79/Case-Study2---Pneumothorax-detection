import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)



# Loading the images
@st.cache
def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    return img
 

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
            st.image(load_image(uploaded_file))
            st.text("Image has been uploaded")
            # To See Details
            st.write(type(uploaded_file))
            # st.write(dir(image_file))
            file_details = {"Filename": uploaded_file.name,
                            "FileType": uploaded_file.type,
                            "FileSize": uploaded_file.size}
            st.write(file_details)


            st.write("")
            
            submit = st.button('Predict')
            if submit:
                
                if prediction == 0:
                    st.write('Congratulation',name,'You are not diabetic')
                else:
                    st.write(name," we are really sorry to say but it seems like you are Diabetic. But don't lose hope we have suggestions for you:")

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
