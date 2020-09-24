import cv2
import numpy as np 
import argparse
import time
import tensorflow as tf
import keras
from helpers.yolo import load_image, load_yolo, args, detect_objects, get_box_dimensions,cropping_image
from model.siamese import model
import os
import pickle

# Load the yolo model
net, classes, colors, output_layers = load_yolo()
# Load Classifier
model.load_weights("weights/weights.8000.h5")
# Load the predicting image
img, height, width, channels = load_image(args.img_path)
# Detect obj
blob, outputs = detect_objects(img, net, output_layers)
boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
crop_image = cropping_image(boxes, confs, img)
crop_img = cv2.cvtColor(cv2.resize(crop_image, (105,105)),cv2.COLOR_BGR2RGB)
# Image_array
with open('model/images_array105.pickle', 'rb') as f:
    images_array = pickle.load(f)
with open('model/X_test105.pickle', 'rb') as f:
    X_test = pickle.load(f)
names = list(X_test.keys())
# Crop image
crop_img = np.expand_dims(crop_img,0)
crop_img = np.tile(crop_img,(images_array.shape[0],1,1,1))
# cv2.imshow('image',crop_image)
# cv2.waitKey()

