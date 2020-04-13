import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots

from datetime import datetime
from imutils import paths
import pandas as pd
import numpy as np
import argparse
import pickle
import cv2
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from architecture import models
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
print(f'Tensorflow version: {tf.__version__}')

def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text='Number of epochs', row=1, col=1)
    fig.update_xaxes(title_text='Number of epochs', row=2, col=1)
    fig.update_xaxes(title_text='Accuracy', row=1, col=1)
    fig.update_xaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f'Metrics')

    po.plot(fig, filename=filename, auto_open=False)


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, help='Path to the data')
ap.add_argument('-e', '--epochs', default=1, help='Numbers of epochs.', type=int)
args = vars(ap.parse_args())

LEARNING_RATE = 0.001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)

print('[INFO] Loading data ...')
image_paths = list(paths.list_images(args['images']))
np.random.shuffle(image_paths)


data = []
labels = []

for image_path in image_paths:
    org_image = cv2.imread(image_path)
    image = cv2.resize(org_image, (150, 150))
    image = img_to_array(image)
    data.append(image)

    label = [image_path.split('\\')[-2]]
    labels.append((label))

data = np.array(data, dtype='float') / 255.
labels = np.array(labels)
print(f'[INFO] {len(image_paths)} images with size: {data.nbytes / (1024 * 1000.0):.2f} MB')
print(f'[INFO] Data shape {data.shape}')

print(f'[INFO] Binarization of labels...')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(f'[INFO] Labels: {mlb.classes_}')

if not os.path.exists(r'./output'): os.mkdir(r'./output')

with open(r'output\mlb.pickle', 'wb') as file:
    file.write(pickle.dumps(mlb))

print('[INFO] Data split...')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
print(f'[INFO] Rozmiar danych treningowych: {X_train.shape}')
print(f'[INFO] Rozmiar danych testowych: {X_test.shape}')

print(f'[INFO] Construction of the generator...')
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print(f'[INFO] Model building ...')
architecture = models.Vgg16(input_shape=INPUT_SHAPE, num_classes=len(mlb.classes_), final_activation='sigmoid')
model = architecture.build()
model.summary()

model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('output', 'model_' + dt + '.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)

print(f'[INFO] Training the model...')
history = model.fit_generator(
    generator=train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint])

filename = os.path.join('output', 'report_' + dt + '.html')
print(f'[INFO] Export the chart to a file {filename}...')
plot_hist(history, filename=filename)

print('[INFO] End')