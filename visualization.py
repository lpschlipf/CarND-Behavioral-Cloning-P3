"""
A script for visualizing some interesting data to help understand the modules.
"""

import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy import ndimage
from sklearn.model_selection import train_test_split

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

    samples = csv_lines
    batch_size = 32
    num_samples = len(samples)
    images = []
    angles = []
    for offset in range(200*batch_size, 210*batch_size, batch_size):
        batch_samples = samples[offset:offset + batch_size]

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

        X_train = np.array(images)
        y_train = np.array(angles)

        print('The number of used samples: {}, that equals a number of images of {}'.format(num_samples, X_train.shape))

    # Visualize some stuff
    # Image preprocessing
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(X_train[30])
    ax2.imshow(X_train[30][50:140]/255.)

    # Sample training images, all three cameras
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    idx = 150
    ax1.imshow(images[idx + 1])
    ax2.imshow(images[idx])
    ax3.imshow(images[idx + 2])

    # Recovery maneuver, 3 images
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    idx = 30
    ax1.imshow(images[idx])
    ax2.imshow(images[idx + 51])
    ax3.imshow(images[idx + 102])
    ax4.imshow(images[idx + 150])

    # Image flipping
    fig, (ax1, ax2) = plt.subplots(1, 2)
    idx = 30
    ax1.set_title('Steering Angle: {:.2f}'.format(augmented_angles[idx]))
    ax1.imshow(augmented_images[idx])
    ax2.set_title('Steering Angle: {:.2f}'.format(augmented_angles[idx+1]))
    ax2.imshow(augmented_images[idx+1])

    # Three camera training
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    idx = 30
    ax1.set_title('Steering Angle: {:.2f}'.format(angles[idx] + 0.3))
    ax1.imshow(images[idx + 1])
    ax2.set_title('Steering Angle: {:.2f}'.format(angles[idx]))
    ax2.imshow(images[idx])
    ax3.set_title('Steering Angle: {:.2f}'.format(angles[idx] - 0.3))
    ax3.imshow(images[idx + 2])

    plt.show()
