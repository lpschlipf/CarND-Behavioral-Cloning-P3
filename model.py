"""
This module includes a deep neural network architecture based on the keras package.
Additionally methods are provided and executed here that take care of data processing, training and evaluating the
performance of the DNN.
"""

import numpy as np
import csv
import os
from scipy import ndimage
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, AveragePooling2D, Lambda, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle
from math import ceil


def create_model_architecture(input_shape=(160, 320, 3)):
    model = Sequential()
    ### Data Preprocessing ###
    # Norming Pixels and mean centering
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # Cropping image to remove unnecesary and confusing horizon and car hood.
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    ### LeNet 5 ###
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


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.3
                angle_left = angle + correction
                angle_right = angle - correction
                angles.extend((angle, angle_left, angle_right))
                # Careful with the path, it was recorded in a workspace under unix, check where you are!
                img_name_center = os.path.split(batch_sample[0])[-1]
                img_name_left = os.path.split(batch_sample[1])[-1]
                img_name_right = os.path.split(batch_sample[2])[-1]
                images.append(ndimage.imread(os.path.join('data_regular/IMG/', img_name_center)))
                images.append(ndimage.imread(os.path.join('data_regular/IMG/', img_name_left)))
                images.append(ndimage.imread(os.path.join('data_regular/IMG/', img_name_right)))

            # Also create flipped images with negated steering angle to gain more data
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(-1.0 * angle)

            # After some experimentation, it turns out that only using non-flipped data works better
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


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

    train_samples, validation_samples = train_test_split(csv_lines, test_size=0.2)

    # Set our batch size
    batch_size = 32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    # Create model architecture
    model = create_model_architecture()

    # Stick it in the optimizer and train it, finally save.
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_samples) / batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples) / batch_size),
                        epochs=7, verbose=1)
    model.save('model.h5')
