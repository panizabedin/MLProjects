from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


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



# one hot encoding of the labels
training_set_labels = to_categorical(training_set_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
validation_set_labels = to_categorical(validation_set_labels, num_classes=10)

print("--------------------First Architecture--------------------")

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


def history(training_data, training_data_labels, validation_data, validation_data_labels, epochs, batch_size):
    history = model.fit(training_data, training_data_labels,
                          epochs=epochs, batch_size=batch_size,
                          validation_data=(validation_data, validation_data_labels))
    return history



def plot(history, epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    test_loss_values = history_dict['val_loss']
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, loss_values, 'bo', label='Training loss')
    plt.plot(epochs_range, test_loss_values, 'ro', label='Test loss')
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

plot(history(training_set, training_set_labels, validation_set,validation_set_labels, 10, 64),10)

print("--------------------Second Architecture--------------------")

