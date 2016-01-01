# import the necessary packages
from colordescriptor import ColorDescriptor
import cv2
import numpy as np
from os import listdir
from os.path import basename, join
from sklearn.cluster import KMeans

def create_features(path):
    # initialize the color descriptor
    cd = ColorDescriptor((4, 4, 2))
     
    # use glob to grab the image paths and loop over them
    image_names, X = [], []
    for imagePath in listdir(path): 
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        imageID = basename(imagePath).split('.')[0] 
        image = cv2.imread(join(path,imagePath))

        # describe the image
        features = cd.describe(image)

        image_names.append(imageID)
        X.append(features)
    return np.array(image_names), np.array(X)

def kmeans(X):
    model = KMeans(n_clusters=3, n_jobs=-1)
    model.fit(X)
    return model

def predict_clusters(model, X):
    return model.predict(X)

