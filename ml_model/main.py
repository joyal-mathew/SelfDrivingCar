import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

# Keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Rescaling, InputLayer
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

# def load_path(path):
#     img = tf.io.read_file(path)
#     img = tf.io.decode_png(img, channels=3)
#     img = tf.image.resize(img, [224, 224])
#     # img = cv2.GaussianBlur(img, (3, 3), 0)
#     # img = cv2.resize(img, (224, 224))
#     # img = np.array(img)
#     img = img[..., ::-1] #todo make sure this should be needed
#     img = img / 255
#     # img = preprocess_input(img)
#     # img = img_to_array(img)
#     return img

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

    short = True;

    dataset = tf.data.Dataset.list_files(path + "input/*", shuffle=False)
    if short:
        dataset = dataset.take(1000)
    print(dataset)

    angles = np.load(path + "output.npy")
    if short:
        angles = angles[:1000]
    # angles_flipped = 1 - angles #todo make sure this is working as intended

    angles = tf.data.Dataset.from_tensor_slices(angles.tolist())
    # angles_flipped = tf.data.Dataset.from_tensor_slices(angles_flipped.tolist())
    # angles_all = angles.concatenate(angles_flipped)

    # if process:
    dataset_unflipped = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset_flipped = dataset.map(process_path_flipped, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset_flipped.concatenate(dataset_flipped)
    dataset = tf.data.Dataset.zip((dataset_unflipped, angles))
    # else:
    #     dataset = dataset.map(load_path, num_parallel_calls=tf.data.AUTOTUNE)
    #     dataset = tf.data.Dataset.zip((dataset, angles))

    if os.name != "nt":
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # print("input images: ", tf.data.experimental.cardinality(dataset).numpy())
    # print(tf.data.experimental.cardinality(val_ds).numpy())

    # for image, label in dataset.take(10):
    #   print("Image shape: ", image.numpy().shape)
    #   print("Label: ", label.numpy())

    # for image, label in dataset.take(10):
    #   print("Image shape: ", image.numpy().shape)
    #   print("Label: ", label.numpy())

    print("== cardinality", dataset.cardinality())
    return dataset

def diversify_data(dataset):

    def map_func(*pair):
        # print("type", type(pair))
        # print("structure", x,y)
        # return tf.data.Dataset.from_tensor_slices([x,y])
        # return ((x,y), (x,y))
        # list = []
        # list.append(tf.data.Dataset.from_tensors([x,y]))
        # list.append(tf.data.Dataset.from_tensors([x,y]))
        # return tf.data.Dataset.from_tensors(pair)
        # return tf.data.Dataset.from_tensor_slices([pair])
        # tf.data.
        return (pair, pair)

    dataset = dataset.map(map_func)
    print (dataset.element_spec)
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    # dataset = dataset.unbatch()

    print("== cardinality", dataset.cardinality())
    print (dataset.element_spec)

    return dataset

batch_size = 128;
if os.name == "nt":
    batch_size = 64
#
input_data = load_data(directory)
# input_train, input_test = split_dataset(input_data, left_size = .1)
input_train = input_data 
input_test = input_data

# input_train = diversify_data(input_train)

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
for layer in mobile_net.layers[:-5]:
    layer.trainable = False
# for layer in mobile_net.layers[:]:
#     layer.trainable = False

# for layer in resnet.layers:
#     print(layer, layer.trainable)

def make_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(224, 224, 3)))

    # model.add(Rescaling(scale=1.0/255.0))
    model.add(Rescaling(scale=1.0/127.5, offset=-2))

    model.add(mobile_net)
    # model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(100, activation='leaky_relu'))
    model.add(Dense(50, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='leaky_relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.add(Dense(1))
    optimizer = Adam(learning_rate=1e-3)
    # optimizer = RMSprop(learning_rate= 1e-5)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = make_model()
print(model.summary())
# print(model.layers)
# model.layers[0].trainable=False 
checkpoint_filepath = '/tmp/checkpoint'

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
          patience=8,
          cooldown=5),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_weights_only=True,
        monitor='loss',
        # mode='max',
        save_best_only=True),

    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]


try:
    # model.fit(input_train, validation_data=input_test, epochs=200, verbose=1, shuffle=1, callbacks=my_callbacks)
    model.fit(input_train, epochs=200, verbose=1, shuffle=1, callbacks=my_callbacks)
except KeyboardInterrupt:
    pass

model.load_weights(checkpoint_filepath)


# history = model.fit(input_data, output_data, epochs=25, batch_size=256, verbose=1, shuffle=1)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
#
# plt.show()

input_images = []
input_images = load_data(directory)
# display_images = load_data(directory, False, False)
# input_images = input_images.map(lambda x,y: x)
input_images = input_images.batch(batch_size)
print("predicting...")
predictions = model.predict(input_images)

if (input("Would you like to preview the model's predictions? y/N\n") == "y"):
# if (True):

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

    for img, prediction, output in zip(input_images, predictions, outputs):
        # input_image = cv2.pyrDown(np.array(input_image[0]))
        img = img[0].numpy() / 255
        # input_image = input_image.reshape((224, 224, 3))
        # print("size: ", input_image)
        draw_line(img, .5, (0, 0, 0), 1)
        draw_line(img, output, (0, 255, 0))
        draw_line(img, prediction, (0, 0, 255))

        img = cv2.resize(img, [500, 500])
        cv2.imshow('Frame', img)
        print(f"{prediction[0]=}, {output=}")

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    print("prediciton stddev: ", np.std(predictions))
    print("prediction mean: ", np.mean(predictions))
    print("truth stddev: ", np.std(outputs))
    print("truth mean: ", np.mean(outputs))

    # todo plot a comparison graph of these two arra
