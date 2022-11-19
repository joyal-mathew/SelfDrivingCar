import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

# Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import load_img
from keras.utils import img_to_array

import cv2
# import pandas as pd
import random
import ntpath

# Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


directory = "annotator/dataset/"

def process_image(img):
    img = cv2.cvtColor(img_to_array(img), cv2.COLOR_RGB2YUV)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (100, 100))
    img = img / 255

    img = img_to_array(img)
    return img

def load_data(dir):

    names = sorted(os.listdir(directory + "input"))

    dataset = []
    for name in names:
        img = load_img(dir + "input/" + name, color_mode = "rgb")
        img = process_image(img)
        dataset.append(img)

    angles = np.load(dir + "output.npy")
    return np.array(dataset), angles

input_data, output_data = load_data(directory)

input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

print("INPUTS")
print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))

print("OUTPUTS:")
print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))

print(input_data.size)
print(output_data.size)

from keras.applications import ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

for layer in resnet.layers[:-4]:
    layer.trainable = False
 
# for layer in resnet.layers:
#     print(layer, layer.trainable)

def make_model():
  model = Sequential()
  model.add(resnet)
  # model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(50, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(10, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(1))
  optimizer = Adam(learning_rate=1e-3)
  model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
  return model

model = make_model()
# print(model.summary())

history = model.fit(input_train, output_train, epochs=25, validation_data=(input_test, output_test), batch_size=128, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

# plt.show()

if (input("Would you like to preview the model's predictions? y/N") == "y"):

