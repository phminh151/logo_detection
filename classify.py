import cv2
# import numpy as np 
# import argparse
import time
# import tensorflow as tf
# import keras
from detection import model, crop_img, images_array, names,crop_image   

# Predicting
all_score = {}
score = model.predict([crop_img,images_array])
for n, i in enumerate(score):
  name = names[n]
  all_score[name] = i[0]
# Return Prediction
n = 0
for w in sorted(all_score, key=all_score.get, reverse=True):
    n+=1
    print(w, all_score[w])
    if n==10:
      break
# Return Image
cv2.imshow('image',crop_image)
cv2.waitKey()