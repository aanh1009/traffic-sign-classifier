import streamlit as st
import pickle
from PIL import Image
import torch
def load_data():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data['data_transforms']
def load_model():
    with open('model_scripted.pt','rb') as f:
        model = torch.jit.load('model_scripted.pt')
    return model
data_transforms = load_data()
model = load_model()

def show_classify():
    st.set_page_config(layout="centered")
    st.title('Traffic Signs Classification')
    st.write("""### Insert a traffic sign to classify""")
    image = st.file_uploader("Choose an image")
    ok = st.button('Start classifying')
    if ok:
        img = Image.open(image)
        st.image(img)
        img = data_transforms(img)
        label = torch.argmax(model(img.view(-1,3,128,128)))
        st.subheader(f"This sign seems to be {label}")
    
    


