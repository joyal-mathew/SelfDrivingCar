import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

# Keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Rescaling, InputLayer
import keras.layers as layers
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import split_dataset

# from keras.applications.resnet import preprocess_input


import cv2
# import pandas as pd
import random

# Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



directory = "annotator/dataset/"

def process_path_flipped(path):
    return process_path(path, True)

def process_path(path, flip = False):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    if flip:
        img = tf.image.flip_left_right(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (224, 224))
    # img = img / 255
    # img = preprocess_input(img)
    # img = img_to_array(img)
    return img

def load_data(path):

    flipped = True
    short = False
    num = 400

    dataset = tf.data.Dataset.list_files(path + "input/*", shuffle=False)
    if short:
        dataset = dataset.take(num)
    print(dataset)

    angles = np.load(path + "output.npy")
    if short:
        angles = angles[:num]
    angles_flipped = (angles * - 1) + 1 #todo make sure this is working as intended
    if short:
        angles_flipped = angles_flipped[:num]


    angles_all = []
    if flipped:
        angles_all = np.concatenate((angles, angles_flipped))
    else:
        angles_all = angles

    angles_all = tf.data.Dataset.from_tensor_slices(angles_all.tolist())

    # angles = tf.data.Dataset.from_tensor_slices(angles.tolist())
    # angles_flipped = tf.data.Dataset.from_tensor_slices(angles_flipped.tolist())
    # angles_all = angles.concatenate(angles_flipped)
    print("angles cardinatility", angles_all.cardinality())

    # if process:
    dataset_unflipped = dataset.map(process_path)
    # dataset_unflipped = dataset.map(process_path)


    dataset_flipped = dataset.map(process_path_flipped)
    dataset = dataset_unflipped.concatenate(dataset_flipped)

    if not flipped:
        dataset = dataset_unflipped

    print("dataset cardinatility", dataset.cardinality())
    dataset = tf.data.Dataset.zip((dataset, angles_all))

    if os.name != "nt":
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("== cardinality", dataset.cardinality())
    return dataset


    return dataset

batch_size = 128;
if os.name == "nt":
    batch_size = 64
#
input_data = load_data(directory)
# input_data = input_data.shuffle(10000)
# input_train, input_test = split_dataset(input_data, left_size = .5)
input_test, input_train = split_dataset(input_data, left_size = .5)

# input_train = load_data(directory) 
# input_test = load_data(directory) 
# input_train = diversify_data(input_train)

input_train = input_train.shuffle(10000, reshuffle_each_iteration=True)

input_train = input_train.batch(batch_size)
input_test = input_test.batch(batch_size)
# input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

# print("INPUTS")
# print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))
#
# print("OUTPUTS:")
# print("Training Samples: {}\nTest Samples: {}".format(len(input_train), len(input_test)))
#
# print(input_data.size)
# print(output_data.size)

# from keras.applications import ResNet50
from keras.applications import MobileNetV2
mobile_net = MobileNetV2(weights='imagenet' ) #include_top=False

# input_data = preprocess_input(input_data)
for layer in mobile_net.layers[:]: #why does setting this to 5 work better than 10
    layer.trainable = False
# for layer in mobile_net.layers[:]:
#     layer.trainable = False

# for layer in resnet.layers:
#     print(layer, layer.trainable)

def make_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(224, 224, 3)))

    model.add(layers.RandomBrightness(factor=(-.3, .3)))
    model.add(layers.RandomContrast(factor=(.2, .2)))

    model.add(layers.RandomTranslation(height_factor=(.2, .2), width_factor=(0.0, 0.0), fill_mode="nearest"))
    # model.add(layers.RandomRotation(factor=(.1, .1), fill_mode="nearest"))
    # model.add(lay)
    # model.add(Rescaling(scale=1.0/255.0))

    model.add(Rescaling(scale=1.0/127.5, offset=-1))

    model.add(mobile_net)
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(100, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='tanh'))
    model.add(Dense(1))
    # model.add(Dense(1))
    optimizer = Adam(learning_rate=1e-3)
    # optimizer = RMSprop(learning_rate= 1e-3)
    # optimizer = SGD(learning_rate = 1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = make_model()
print(model.summary())
# print(model.layers)
# model.layers[0].trainable=False 
checkpoint_filepath = './tmp/checkpoint'

my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(
    #     patience=5,
    #     verbose=True,
    #     monitor="loss",
    #     restore_best_weights=True),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        verbose=True,
        min_lr=1e-4,
          patience=8,
          cooldown=5),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=0,
        save_weights_only=True,
        monitor='loss',
        # mode='max',
        save_best_only=True),

    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

#Train the dense layers and part of the model
mobile_net.trainable = True;
for layer in mobile_net.layers[:-5]: #why does setting this to 5 work better than 10
    layer.trainable = False

optimizer = Adam(learning_rate=1e-3)
# optimizer = SGD(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimizer)
print(model.summary())


try:
    # model.fit(input_train, epochs=200, verbose=1, callbacks=my_callbacks)
    model.fit(input_train, validation_data=input_test, epochs=200, verbose=1, callbacks=my_callbacks)
except KeyboardInterrupt:
    pass

model.load_weights(checkpoint_filepath)



input_images = []
input_images = load_data(directory)
# display_images = load_data(directory, False, False)
# input_images = input_images.map(lambda x,y: x)
input_images = input_images.batch(batch_size)
print(f"predicting {input_images.cardinality()} batches...")
predictions = model.predict(input_images)

input_images = input_images.unbatch()
# print(predictions.element_spec)
print(f"predicted {input_images.cardinality()} images")
print(predictions)

# outputs = np.load(directory + "output.npy")[:400]
# print(outputs)
# outputs_flipped = (outputs * - 1) + 1 #todo make sure this is working as intended
# print(outputs_flipped)

# outputs = np.concatenate((outputs, outputs_flipped))

if (input("Would you like to preview the model's predictions? y/N\n") == "y"):
# if (True):



    def draw_line(img, rel_x, color, thickness = 2):
        print (img.size)
        abs_x = 224 * rel_x

        cv2.line(img, (int(abs_x),0), (int(abs_x), 224), color, thickness = thickness)

    for i, (data, prediction) in enumerate(zip(input_images, predictions)):
        # print(output)
        # print(img)
        img, true_angle = data
        # print(in_angle)
        # print(type(in_angle))
        # input_image = cv2.pyrDown(np.array(input_image[0]))
        img = img.numpy() / 255
        # img = img[..., ::-1] #todo make sure this should be needed
        # input_image = input_image.reshape((224, 224, 3))
        # print("size: ", input_image)
        draw_line(img, .5, (0, 0, 0), 1)
        draw_line(img, true_angle, (0, 255, 0))
        if (i < len(predictions)/2):
            draw_line(img, prediction, (0, 0, 255))
        else:
            draw_line(img, prediction, (2500, 0, 0))

        img = cv2.resize(img, [500, 500])
        cv2.imshow('Frame', img)
        print(f"{prediction[0]=}, {true_angle=}")

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    print("prediciton stddev: ", np.std(predictions))
    print("prediction mean: ", np.mean(predictions))
    # print("truth stddev: ", np.std(outputs))
    # print("truth mean: ", np.mean(outputs))

    # todo plot a comparison graph of these two arra
