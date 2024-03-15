# import streamlit as st
# import os
# from PIL import Image
# import numpy as np
# import pickle
# import tensorflow
# from tensorflow import keras
# from keras.applications.resnet50 import ResNet50,preprocess_input
# from keras.layers import GlobalMaxPooling2D
# from keras.preprocessing import image

# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm

# feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))

# model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# model.trainable = False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# st.title('Fashion Recommender System')

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return 1
#     except:
#         return 0

# def feature_extraction(img_path,model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)

#     return normalized_result

# def recommend(features,feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)

#     distances, indices = neighbors.kneighbors([features])

#     return indices

# # steps
# # file upload -> save
# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # display the file
#         display_image = Image.open(uploaded_file)
#         st.image(display_image)
#         # feature extract
#         features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
#         #st.text(features)
#         # recommendention
#         indices = recommend(features,feature_list)
#         # show
#         col1,col2,col3,col4,col5 = st.beta_columns(5)

#         with col1:
#             st.image(filenames[indices[0][0]])
#         with col2:
#             st.image(filenames[indices[0][1]])
#         with col3:
#             st.image(filenames[indices[0][2]])
#         with col4:
#             st.image(filenames[indices[0][3]])
#         with col5:
#             st.image(filenames[indices[0][4]])
#     else:
#         st.header("Some error occured in file upload")



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
    normilized = result / norm(result)
    
    distance, indices = NearestNeighbors.kneighbors([normilized])
    similar_images = []
    for file in indices[0][1:6]:
        similar_images.append(cv2.imread(filenames[file]))

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

