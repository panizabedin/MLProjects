from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
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
validation_set_size = int(0.2 * 50000)

validation_set = train_images_shuffled[0:validation_set_size]
validation_set_labels = train_labels_shuffled[0:validation_set_size]

training_set = train_images_shuffled[validation_set_size:]
training_set_labels = train_labels_shuffled[validation_set_size:]




# one hot encoding of the labels
train_labels_shuffled = to_categorical(train_labels_shuffled, num_classes=10)
training_set_labels = to_categorical(training_set_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
validation_set_labels = to_categorical(validation_set_labels, num_classes=10)


# training function
def history(model, training_data, training_data_labels, validation_data, validation_data_labels, epochs, batch_size):
    history = model.fit(training_data, training_data_labels,
                          epochs=epochs, batch_size=batch_size,
                          validation_data=(validation_data, validation_data_labels))
    return history


# plotting loss and accuracy corresponding to the validation set
def plot_validation(history, epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    test_loss_values = history_dict['val_loss']
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, loss_values, 'bo', label='Training loss')
    plt.plot(epochs_range, test_loss_values, 'ro', label='validation')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc_values = history_dict['acc']
    test_acc_values = history_dict['val_acc']
    plt.plot(epochs_range, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs_range, test_acc_values, 'ro', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# plotting loss and accuracy corresponding to the test set
def plot_test(history, epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    test_loss_values = history_dict['val_loss']
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, loss_values, 'bo', label='non-test loss')
    plt.plot(epochs_range, test_loss_values, 'ro', label='test loss')
    plt.title('non-test and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc_values = history_dict['acc']
    test_acc_values = history_dict['val_acc']
    plt.plot(epochs_range, acc_values, 'bo', label='non-test accuracy')
    plt.plot(epochs_range, test_acc_values, 'ro', label='test accuracy')
    plt.title('non-test and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
#plot(history(training_set, training_set_labels, validation_set,validation_set_labels, 10, 64),10)

print("--------------------First Architecture--------------------")
# model = models.Sequential()
# #layers
# #1
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# #2
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# #3
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
# model.add(layers.Flatten())
# model. add(layers.Dense(256, activation='relu'))
# model. add(layers.Dense(10, activation='softmax'))
#
# model.summary()
#
#
# model.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])



#plot_validation(history(model,training_set, training_set_labels, validation_set,validation_set_labels,10, 64),10)

# print("--------------------Second Architecture with Dropout--------------------")
#
# model_2 = models.Sequential()
# model_2.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model_2.add(layers.MaxPooling2D((2, 2)))
# model_2.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model_2.add(layers.MaxPooling2D((2, 2)))
# model_2.add(layers.Conv2D(32, (3, 3), activation='relu'))
# #
# model_2.add(layers.Flatten())
# # DROPOUT
# model_2.add(layers.Dropout(0.5))
# model_2.add(layers.Dense(512, activation='relu'))
# model_2.add(layers.Dense(10, activation='softmax'))
#
# model_2.summary()
#
#
# model_2.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])

# print("--------------------Third Architecture with data augmentation--------------------")
#
# model_3 = models.Sequential()
# #layers
# #1
# model_3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model_3.add(layers.MaxPooling2D((2, 2)))
# #2
# model_3.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model_3.add(layers.MaxPooling2D((2, 2)))
# #3
# model_3.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model_3.add(layers.MaxPooling2D(2, 2))
#
# model_3.add(layers.Flatten())
# model_3.add(layers.Dense(64, activation='relu'))
# model_3.add(layers.Dense(10, activation='softmax'))
#
# model_3.summary()
#
#
# model_3.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
#
# train_datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True)
#
# validation_datagen = ImageDataGenerator()
#
# train_generator = train_datagen.flow(training_set, training_set_labels, batch_size=32)
# validation_generator = validation_datagen.flow(validation_set, validation_set_labels, batch_size=32)
# history_dataaug_model3 = model_3.fit_generator(train_generator, validation_data=validation_generator,
#                                 validation_steps=len(training_set) / 32, steps_per_epoch=len(training_set) / 32,
#                                 epochs=30, verbose=2)


# print("--------------------fourth Architecture with dropout and strides--------------------")
#
# model_4 = models.Sequential()
# model_4.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
# model_4.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
# model_4.add(layers.Conv2D(64, (3, 3), activation = 'relu', strides = 2))
# model_4.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
# model_4.add(layers.Conv2D(128, (3, 3), activation = 'relu', strides=2))
# model_4.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
# model_4.add(layers.Flatten())
# model_4.add(layers.Dropout(0.5))
# model_4.add(layers.Dense(128, activation = 'relu'))
# model_4.add(layers.normalization.BatchNormalization())
# model_4.add(layers.Dense(10, activation = 'softmax'))
#
# model_4.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])

#print("--------------------Fifth Architecture with data augmentation and dropout-------------------")

# model_5 = models.Sequential()
# #CNN layers
# #1
# model_5.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
# #2
# model_5.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
# #3
# model_5.add(layers.Conv2D(256, (3, 3), activation = 'relu', strides = 2))
# #4
# model_5.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
# #5
# model_5.add(layers.Conv2D(512, (3, 3), activation = 'relu', strides=2))
# #6
# model_5.add(layers.Conv2D(512, (3, 3), activation = 'relu'))
#
# #Dense layers
# model_5.add(layers.Flatten())
# #Dropout
# model_5.add(layers.Dropout(0.5))
# model_5.add(layers.Dense(256, activation = 'relu'))
# #batch normalization added
# model_5.add(layers.normalization.BatchNormalization())
# model_5.add(layers.Dense(10, activation = 'softmax'))
#
# model_5.summary()
#
#
# model_5.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
#
# train_datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True)
# validation_datagen = ImageDataGenerator()
#
# train_generator = train_datagen.flow(training_set, training_set_labels, batch_size=32)
# validation_generator = validation_datagen.flow(validation_set, validation_set_labels, batch_size=32)
# history_dataaug_model5 = model_5.fit_generator(train_generator, validation_data=validation_generator,
#                                 validation_steps=len(training_set) / 32, steps_per_epoch=len(training_set) / 32,
#                                 epochs=30, verbose=2)'

# training the data on both training and validation set with the best architecture above (arch 5), then testing on the test data
print("--------------------Winner : Fifth Architecture with data augmentation and dropout-------------------")

model_winner = models.Sequential()
#CNN layers
#1
model_winner.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
#2
model_winner.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
#3
model_winner.add(layers.Conv2D(256, (3, 3), activation = 'relu', strides = 2))
#4
model_winner.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
#5
model_winner.add(layers.Conv2D(512, (3, 3), activation = 'relu', strides=2))
#6
model_winner.add(layers.Conv2D(512, (3, 3), activation = 'relu'))


#Dense layers
model_winner.add(layers.Flatten())
model_winner.add(layers.Dropout(0.5))
model_winner.add(layers.Dense(256, activation = 'relu'))
model_winner.add(layers.normalization.BatchNormalization())
model_winner.add(layers.Dense(10, activation = 'softmax'))

model_winner.summary()


model_winner.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

train_datagen2 = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)

validation_datagen2 = ImageDataGenerator()

train_generator2 = train_datagen2.flow(train_images_shuffled, train_labels_shuffled, batch_size=32)
validation_generator2= validation_datagen2.flow(test_images, test_labels, batch_size=32)

history_dataaug_model = model_winner.fit_generator(
    train_generator2,
    steps_per_epoch=30,
    epochs=30,
    validation_data=validation_generator2,
    validation_steps=50
)


train_generator2 = train_datagen2.flow(train_images_shuffled, train_labels_shuffled, batch_size=32)
validation_generator2 = validation_datagen2.flow(test_images, test_labels, batch_size=32)
history_dataaug_model = model_winner.fit_generator(train_generator2, validation_data=validation_generator2,
                                validation_steps=len(train_images_shuffled) / 32, steps_per_epoch=len(train_images_shuffled) / 32,
                                epochs=30, verbose=2)

#model_5 results
plot_test(history_dataaug_model,30)


#score = model.evaluate(validation_set, validation_set_labels, batch_size=128, verbose=0)



#history_data_aug(validation_set, training_set)

# model_2 results
#plot_validation(history(model_2,training_set, training_set_labels, validation_set,validation_set_labels,20, 64),20)

# model_4 results
#plot_validation(history(model_4,training_set, training_set_labels, validation_set,validation_set_labels,10, 64),10)
#score = model_4.evaluate(validation_set, validation_set_labels, batch_size=64, verbose=0)


#model_3 results
#plot_validation(history_dataaug_model3,30)

#print(score)
#plot_test(history(train_images_shuffled, train_labels_shuffled, test_images,test_labels,50, 64),50)

#score = model.evaluate(training_set, y_test, batch_size=128)



