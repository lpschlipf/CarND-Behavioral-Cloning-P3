"""
This module includes a deep neural network architecture based on the keras package.
Additionally methods are provided and executed here that take care of data processing, training and evaluating the
performance of the DNN.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy import ndimage
from keras.applications.mobilenetv2 import mobilenet_v2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, AveragePooling2D


def create_model_architecture(input_shape=(160, 320, 3)):
    model = Sequential()
    # Convolutional Layer 1
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D())
    # Convolutional Layer 2
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    # Flatten Layer
    model.add(Flatten())
    # Dropout
    model.add(Dropout(0.2))
    # Fully connected 1
    model.add(Dense(units=120, activation='relu'))
    # Fully connected 2
    model.add(Dense(units=84, activation='relu'))
    # Fully connected 3 - Gives the steering angle
    model.add(Dense(units=1))
    return model

if __name__ == '__main__':
    # Data readin
    csv_lines = []
    images = []
    angles = []
    # Get lines from all csv files
    for file in os.listdir('data_regular'):
        if file.endswith('.csv'):
            with open(os.path.join('data_regular', file)) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    csv_lines.append(line)
    # read images and steering angles based on what is written in lines.
    for line in csv_lines:
        angles.append(float(line[3]))
        # Careful with the path, it was recorded in a workspace under unix, check where you are!
        img_name = os.path.split(line[0])[-1]
        images.append(ndimage.imread(os.path.join('data_regular/IMG/', img_name)))

    # Finalize training data
    X_train = np.array(images)
    y_train = np.array(angles)

    # Create model architecture
    model = create_model_architecture()

    # Stick it in the optimizer and train it
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
    model.save('model.h5')
