from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import imutils
import argparse
import cv2
import os

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG__LEVEL'] = '3'

def load(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (150,150))
    image = image.astype('float') / 255.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to image')
ap.add_argument('-m', '--model', required=True, help='Path to model')
args = vars(ap.parse_args())

print('[INFO] Loading model ...')
model = load_model(args['model'])
image =load(args['image'])
y_pred = model.predict(image)[0]

print('[INFO] Loading label ...')
with open('output\mlb.pickle', 'rb') as file:
    mlb = pickle.loads(file.read())

labels = dict(enumerate(mlb.classes_))
idx = np.argsort(y_pred)[-1]

print('[INFO] Loading image ...')
image = cv2.imread(args['image'])
image = imutils.resize(image, width=500)

print('[INFO] Displaying image ...')
cv2.putText(img=image, text=f'{labels[idx]}',
            org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(205, 133, 63), thickness=2)

cv2.imshow('image', image)
cv2.waitKey(0)
print('ok')
