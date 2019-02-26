import os
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
import cv2


ROOT = os.getcwd()
IMAGE_SIZE = 150
LEARNING_RATE = 1e-3
MODELS_PATH = os.path.join(ROOT, 'model')
ONE_HOT = np.array(['American Shorthair', 'Angora', 'Ashera', 'British Shorthair',
                    'Exotic', 'Himalayan', 'Maine Coon', 'Persian', 'Ragdoll', 'Siamese', 'Sphynx'])


def load_custom_structure(classes=11):
    convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 512, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 1024, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, classes, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=2)

    return model


def load_alex_structure():
    network = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')

    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = local_response_normalization(network)

    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = local_response_normalization(network)

    network = conv_2d(network, 384, 3, activation='relu')

    network = conv_2d(network, 384, 3, activation='relu')

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 17, activation='softmax')
    network = regression(network, optimizer='momentum', name='targets',
                         loss='categorical_crossentropy', learning_rate=LEARNING_RATE)

    model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=2)


def load_custom_values(model, model_name):
    # model_name = '<name>.model', dtype=string
    dir_name = model_name.split('.')[0]
    model_dir_path = os.path.join(MODELS_PATH, dir_name)
    model_values_path = os.path.join()

    model.load
