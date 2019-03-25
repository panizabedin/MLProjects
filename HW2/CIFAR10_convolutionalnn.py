from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)

print(test_images.shape)

train_images = train_images.reshape((50000, 32, 32, 3)) / 255.0
test_images = test_images.reshape((10000, 32, 32, 3)) / 255.0

m = train_images.shape[0]
shuffled_indices = np.random.permutation(m)
train_images_shuffled = train_images[shuffled_indices]
train_labels_shuffled = train_labels[shuffled_indices]

validation_set_size = int(0.2 * 50000)

validation_set = train_images_shuffled[0:validation_set_size]
validation_set_labels = train_labels_shuffled[0:validation_set_size]

training_set = train_images_shuffled[validation_set_size:]
training_set_labels = train_labels_shuffled[validation_set_size:]


# Examples of training set with their labels
# cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print('Example training images and their labels: ' + str([x[0] for x in training_set_labels[0:5]]))
# print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in training_set_labels[0:5]]))
#
# f, axarr = plt.subplots(1, 5)
# f.set_size_inches(16, 6)
#
# for i in range(5):
#     img = training_set[i]
#     axarr[i].imshow(img)
# plt.show()

print(training_set_labels[0])

print(validation_set.shape)
print(training_set.shape)


# one hot encoding of the labels
training_set_labels = to_categorical(training_set_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
validation_set_labels = to_categorical(validation_set_labels, num_classes=10)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    #
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    #
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

epochs = 2
history = model.fit(training_set,
                      training_set_labels,
                      epochs=epochs,
                      validation_data=(validation_set, validation_set_labels))


def plotLosses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

plotLosses(history)
