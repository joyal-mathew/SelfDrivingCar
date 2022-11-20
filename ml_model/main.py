import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

# Keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import load_img
from keras.utils import img_to_array

from keras.applications.resnet import preprocess_input

import cv2
# import pandas as pd
import random
import ntpath

# Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


directory = "annotator/dataset/"

def load_path(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (224, 224))
    # img = np.array(img)
    img = img[..., ::-1]
    img = img / 255
    # img = preprocess_input(img)
    # img = img_to_array(img)
    return img

def process_path(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (224, 224))
    # img = img / 255
    img = preprocess_input(img)
    # img = img_to_array(img)
    return img

def load_data(path, batch_size = 512, shuffle = True, process = True):


    dataset = tf.data.Dataset.list_files(path + "input/*", shuffle)
    print(dataset)

    angles = np.load(path + "output.npy")
    angles = tf.data.Dataset.from_tensor_slices(angles.tolist())
    # angles = iter(angles)

    if process:
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(load_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((dataset, angles))


    dataset = dataset.cache()
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # print("input images: ", tf.data.experimental.cardinality(dataset).numpy())
    # print(tf.data.experimental.cardinality(val_ds).numpy())

    # for image, label in dataset.take(10):
    #   print("Image shape: ", image.numpy().shape)
    #   print("Label: ", label.numpy())

    # for image, label in dataset.take(10):
    #   print("Image shape: ", image.numpy().shape)
    #   print("Label: ", label.numpy())

    return dataset
    # for (i, name) in enumerate(names):
    #     # if i == n:
    #     #     break
    #     img = cv2.imread(path + "input/" + name)
    #     if process:
    #         img = process_image(img)
    #     # else:
    #         # img = img
    #         # img /= 255
    #     dataset.append(img)
    #
    # return np.array(dataset), angles

input_data = load_data(directory)

# input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

# print("INPUTS")
# print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))
#
# print("OUTPUTS:")
# print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))
#
# print(input_data.size)
# print(output_data.size)

from keras.applications import ResNet50
resnet = ResNet50(weights='imagenet')

# input_data = preprocess_input(input_data)
for layer in resnet.layers[:-4]:
    layer.trainable = False
# for layer in resnet.layers[:]:
#     layer.trainable = False

# for layer in resnet.layers:
#     print(layer, layer.trainable)

def make_model():
  model = Sequential()
  model.add(resnet)
  # model.add(Dropout(0.5))
  model.add(Flatten())
  # model.add(BatchNormalization())
  model.add(Dense(100, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(50, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(10, activation='elu'))
  # model.add(Dropout(0.5))
  model.add(Dense(1))
  # model.add(Dense(1))
  # optimizer = Adam(learning_rate=1e-3)
  model.compile(loss='mse', optimizer='adam')
  return model

model = make_model()
print(model.summary())

# history = model.fit(input_train, output_train, epochs=10, validation_data=(input_test, output_test), batch_size=128, verbose=1, shuffle=1)
history = model.fit(input_data, epochs=10, batch_size=64, verbose=1, shuffle=1)
# history = model.fit(input_data, output_data, epochs=25, batch_size=256, verbose=1, shuffle=1)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
#
# plt.show()

# if (input("Would you like to preview the model's predictions? y/N\n") == "y"):
if (True):

    input_images = []
    input_images = load_data(directory, 512, False, False)
    # processed_images = input_data
    # names = sorted(os.listdir(directory + "input"))
    # for name in names:
        # img = cv2.imread(directory + "input/" + name)
        # cv2.imshow("test3", img)
        # cv2.waitKey(0)
        # break
        # input_images.append(np.array(img))
    #     img = process_image(img)
    #     processed_images.append(img)


    print("predicting...")
    predictions = model.predict(input_images)
    # predictions = predictions.unbatch()
    # print(predictions.size)
    # predictions = predictions.reshape((-1, *predictions.shape[-1:]))
    input_images = input_images.unbatch()
    print(input_images)
    outputs = np.load(directory + "output.npy")


    def draw_line(img, rel_x, color, thickness = 2):
        print (img.size)
        abs_x = 224 * rel_x

        cv2.line(img, (int(abs_x),0), (int(abs_x), 224), color, thickness = thickness)

    for input_image, prediction, output in zip(input_images, predictions, outputs):
        # input_image = cv2.pyrDown(np.array(input_image[0]))
        input_image = input_image[0].numpy()
        # input_image = input_image.reshape((224, 224, 3))
        # print("size: ", input_image)
        draw_line(input_image, .5, (0, 0, 0), 1)
        draw_line(input_image, output, (0, 255, 0))
        draw_line(input_image, prediction, (0, 0, 255))

        cv2.imshow('Frame', input_image)
        print(f"{prediction[0]=}, {output=}")

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    print("prediction mean: ", np.mean(predictions))
    print("truth mean: ", np.mean(outputs))

    # todo plot a comparison graph of these two arrays
