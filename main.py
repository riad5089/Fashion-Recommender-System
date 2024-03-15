import streamlit as st
import os
import cv2
import numpy as np
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from PIL import Image

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Create Nearest Neighbors model
NearestNeighbors = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='brute')
NearestNeighbors.fit(feature_list)

def recommend_similar_images(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    
    distance, indices = NearestNeighbors.kneighbors([normalized])
    similar_images = []
    for file_index in indices[0][1:6]:
        # Directly use the image array obtained from filenames list
        similar_images.append(cv2.cvtColor(cv2.imdecode(np.fromfile(filenames[file_index], dtype=np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

    return similar_images

st.title('Fashion Recommender System')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Recommend Similar Images'):
        similar_images = recommend_similar_images(img)

        cols = st.columns(5)
        for i, image in enumerate(similar_images):
            cols[i].image(image, caption=f'Similar Image {i+1}', use_column_width=True)


