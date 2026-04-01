import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# load the trained model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('pneumonia_model.pth', map_location='cpu'))
    model.eval()
    return model

# prepare image for the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0) # add batch dimension

# page setup
st.title('Pneumonia Detection from Chest X-rays')
st.write('Upload a chest X-ray image to check for signs of pneumonia')

st.warning('This tool is for educational purposes only and is not a substitute for professional medical diagnosis')

# file uploader
uploaded_file = st.file_uploader(
    'Upload chest X-ray image',
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # show the uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Upload X-ray')
        st.image(image, use_container_width=True)

    with col2:
        st.subheader('Analysis Result')

        # load model and make prediction
        model = load_model()
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            normal_prob = probabilities[0].item()
            pneumonia_prob = probabilities[1].item()

        # show result
        if pneumonia_prob >= 0.5:
            st.error(f'Pneumonia Detected')
        else:
            st.success(f'Normal')

        # show probabilities
        st.write('**Confidence Scores:**')
        st.write(f'Normal: {normal_prob:.1%}')
        st.progress(normal_prob)

        st.write(f'Pneumonia: {pneumonia_prob:.1%}')
        st.progress(pneumonia_prob)

    # show what to look for
    st.subheader('What the model looks for')
    st.write('''
             - **Normal X-ray** - lungs appear mostly dark and clear
             - **Pneumonia X-ray** - white or grey patches indicating fluid or infection
             ''')