import tensorflow
from tensorflow import keras
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))
# print(np.array(feature_list).shape)
filenames=pickle.load(open('filenames.pkl','rb'))   


model = ResNet50(weights="imagenet",include_top=False, input_shape=(224,224,3))
model.trainable=False
model=keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()



img = cv2.imread("download (1).jpg")
img=cv2.resize(img,(224,224))
img=np.array(img)
expand_img=np.expand_dims(img,axis=0)
pre_img=preprocess_input(expand_img)
result=model.predict(pre_img).flatten()
normilized=result/norm(result)
# return normilized

NearestNeighbors=NearestNeighbors(n_neighbors=6,metric='euclidean',algorithm='brute')
NearestNeighbors.fit(feature_list)

distance,indices = NearestNeighbors.kneighbors([normilized])

for file in indices[0][1:6]:
    # print(filenames[file])
    img_name=cv2.imread(filenames[file])
    cv2.imshow('Frame',cv2.resize(img_name,(640,480)))
    cv2.waitKey(0)