import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

model_cnn = load_model("cnn_model_ohe.h5")
model_vgg16 = load_model("vgg16_model_ohe.h5")


def preprocess_cnn_image(img):
    img = img.convert("L").resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, -1)
    return img


def preprocess_vgg16_image(img):
    img = img.convert("RGB").resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(model, img, vgg16=False):
    if vgg16:
        img = preprocess_vgg16_image(img)
    else:
        img = preprocess_cnn_image(img)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions, axis=1)[0]
    return class_idx, predictions[0]


st.title("Image Classification with Neural Networks")

model_choice = st.selectbox("Choose a model", ["CNN Model", "VGG16 Model"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if model_choice == "CNN Model":
            class_idx, predictions = predict(model_cnn, image)
        else:
            class_idx, predictions = predict(model_vgg16, image, True)

        classes = [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        st.write(f"Predicted Class: {classes[class_idx]}")

        st.write("Class Probabilities:")
        for i, prob in enumerate(predictions):
            st.write(f"{classes[i]}: {prob:.4f}")
