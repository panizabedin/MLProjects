from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
from time import time as t
from keras.preprocessing.image import ImageDataGenerator

# loading CIFAR10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# normalizing data
train_images = train_images.reshape((50000, 32, 32, 3)) / 255.0
test_images = test_images.reshape((10000, 32, 32, 3)) / 255.0

# m is the number of training examples
m = train_images.shape[0]
shuffled_indices = np.random.permutation(m)
train_images_shuffled = train_images[shuffled_indices]
train_labels_shuffled = train_labels[shuffled_indices]

# setting 0.20 of the training data to validation set randomly


# one hot encoding of the labels
train_labels_shuffled = to_categorical(train_labels_shuffled, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)


def fold(model, training_set, training_labels, validation_set, validation_set_labels):
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,  # randomly shift images horizontally
        height_shift_range=0.2,  # randomly shift images vertically
        shear_range=0.2,        # randomly shear images
        horizontal_flip=True)  # flip images horizontally

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(training_set, training_labels, batch_size=32)
    validation_generator = validation_datagen.flow(validation_set, validation_set_labels, batch_size=32)
    history_dataaug_model5 = model.fit_generator(train_generator, validation_data=validation_generator,
                                                 validation_steps=len(training_set) / 32,
                                                 steps_per_epoch=len(training_set) / 32,
                                                 epochs=20, verbose=2)

    score = model.evaluate(validation_set, validation_set_labels, batch_size=32, verbose=0)

    return score[1]


def k_fold_validation(training_set, training_labels, n):
    scores = []

    model_5 = models.Sequential()
    model_5.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_5.add(layers.Conv2D(64, (3, 3), activation='relu', strides=2))
    model_5.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_5.add(layers.Conv2D(128, (3, 3), activation='relu', strides=2))
    model_5.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model_5.add(layers.Flatten())

    model_5.add(layers.Dropout(0.5))
    model_5.add(layers.Dense(64, activation='relu'))
    model_5.add(layers.normalization.BatchNormalization())
    model_5.add(layers.Dense(10, activation='softmax'))

    model_5.summary()

    model_5.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    fold_size = math.floor(len(training_set) / n)
    for i in range(n):
        print("---------- Fold", i+1, "----------")
        fold_validation_set = training_set[i * fold_size:i * fold_size + fold_size]
        fold_validation_set_labels = training_labels[i * fold_size:i * fold_size + fold_size]
        fold_training_set = np.delete(training_set, slice(i * fold_size, i * fold_size + fold_size), axis=0)
        fold_training_labels = np.delete(training_labels, slice(i * fold_size, i * fold_size + fold_size), axis=0)
        s = fold(model_5, fold_training_set, fold_training_labels, fold_validation_set, fold_validation_set_labels)

        scores.append(s)

    print(scores)

    scores_average = np.average(scores)
    print("Average scores =", scores_average)


k_fold_validation(train_images_shuffled, train_labels_shuffled, 5)


