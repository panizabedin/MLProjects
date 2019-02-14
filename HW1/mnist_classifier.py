from keras.datasets import mnist
import numpy as np
from pythonds.basic.stack import Stack
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

from keras.utils import to_categorical
from keras import models
from keras import layers


def show_image(image):
    plt.imshow(image.reshape(28, 28))
    plt.grid(None)
    plt.show()


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s


def classify(weight, bias, input_image):
    activation = sigmoid(np.dot(weight.T, input_image.T) + bias)
    return activation


def binary_classify(weight, bias, input_image):
    predicts = classify(weight, bias, input_image)
    print(predicts.shape)
    for i in range(predicts.shape[1]):
        if predicts.T[i] > 0.5:
            predicts.T[i] = 1
        else:
            predicts.T[i] = 0
    return predicts


def ten_digits_classifier(weights, biases, input_image):
    if input_image.ndim < 2:
        number_of_inputs = 1
    else:
        number_of_inputs = len(input_image)

    probabilities = np.zeros((len(weights), number_of_inputs))

    predictions = np.zeros(number_of_inputs)

    for i in range(0, len(weights)):
        probabilities[i] = classify(weights[i], biases[i], input_image)

    for i in range(0, number_of_inputs):
        predictions[i] = np.argmax(probabilities.T[i])

    return predictions


def logistic_regression_batch(train_images, train_labels, epochs, lr, target_label):
    for i in range(len(train_images)):
        if train_labels[i] == target_label:
            train_labels[i] = 1
        else:
            train_labels[i] = 0

    # m is the number of examples
    m = train_images.shape[0]
    # n is the number of features
    n = train_images.shape[1]
    np.random.seed(42)
    weight = np.random.randn(n, 1)
    bias = np.random.randn(1, 1)

    for epoch in range(epochs):
        z = np.dot(weight.T, train_images.T) + bias
        activation = sigmoid(z)
        gradient_w = 1.0 / m * np.dot(train_images.T, (activation - train_labels).T)
        gradient_b = 1.0 / m * np.sum(activation - train_labels)
        weight = weight - lr * gradient_w
        bias = bias - lr * gradient_b

    return weight, bias


def logistic_regression_mini_batch_sgd_loss_ce(train_images, train_labels, epochs, lr, target_label, batch_size):
    modified_train_labels = np.zeros(len(train_labels))
    for i in range(len(train_images)):
        if train_labels[i] == target_label:
            modified_train_labels[i] = 1

    # m is the number of examples
    m = train_images.shape[0]
    # n is the number of features
    n = train_images.shape[1]
    # np.random.seed(42)
    weight = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        train_images_shuffled = train_images[shuffled_indices]
        train_labels_shuffled = modified_train_labels[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = train_images_shuffled[i:i + batch_size]
            yi = train_labels_shuffled[i:i + batch_size]
            z = np.dot(weight.T, xi.T) + bias
            activation = sigmoid(z)
            gradient_w = 1.0 / batch_size * np.dot(xi.T, (activation - yi).T)
            gradient_b = 1.0 / batch_size * np.sum(activation - yi)
            weight = weight - lr * gradient_w
            bias = bias - lr * gradient_b

    return weight, bias


def logistic_regression_mini_batch_sgd_loss_se(train_images, train_labels, epochs, lr, target_label, batch_size):
    modified_train_labels = np.zeros(len(train_labels))
    for i in range(len(train_images)):
        if train_labels[i] == target_label:
            modified_train_labels[i] = 1

    # m is the number of examples
    m = train_images.shape[0]
    # n is the number of features
    n = train_images.shape[1]
    # np.random.seed(42)
    weight = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        train_images_shuffled = train_images[shuffled_indices]
        train_labels_shuffled = modified_train_labels[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = train_images_shuffled[i:i + batch_size]
            yi = train_labels_shuffled[i:i + batch_size]
            z = np.dot(weight.T, xi.T) + bias
            activation = sigmoid(z)

            b = (activation - yi) * activation * (1 - activation)
            gradient_w = 1.0 / batch_size * np.dot(xi.T, b.T)
            gradient_b = 1.0 / batch_size * np.sum(b)
            weight = weight - lr * gradient_w
            bias = bias - lr * gradient_b

    return weight, bias


def calculate_accuracy(weight, bias, test_images, test_labels, target_label):
    false_positive_count = 0
    false_negative_count = 0
    correct_count = 0

    test_predictions = classify(weight, bias, test_images)

    for i in range(0, len(test_images)):
        if test_predictions.T[i] > .5:
            prediction = 1
        else:
            prediction = 0

        if target_label == test_labels[i]:
            if prediction == 1:
                correct_count += 1
            else:
                false_negative_count += 1
        else:
            if prediction == 1:
                false_positive_count += 1
            else:
                correct_count += 1
    return correct_count / len(test_labels)


def ten_digits_accuracy(weights, biases, test_images, test_labels):
    test_predictions = ten_digits_classifier(weights, biases, test_images)

    incorrect_count = np.count_nonzero(test_labels - test_predictions)

    return 1.0 - incorrect_count / len(test_labels)


def new_accuracy(weight, bias, test_images, test_labels, target_label):
    modified_test_labels = np.zeros(len(test_labels))
    for j in range(len(test_labels)):
        if test_labels[j] == target_label:
            modified_test_labels[j] = 1

    test_predictions = binary_classify(weight, bias, test_images)
    test_accuracy = 100 - np.mean(np.abs(test_predictions - modified_test_labels) * 100)
    return test_accuracy


def mapping_grey_scale_to_01(input_images):
    return np.ceil(input_images)


def number_of_regions_4_ways(input_image):
    rounded_input_images = mapping_grey_scale_to_01(input_image)

    black_pixels_indices = set(np.nonzero(rounded_input_images)[0])
    white_pixels_indices = set(range(0, 28 * 28)) - black_pixels_indices

    visited = black_pixels_indices
    component_count = 0
    while len(white_pixels_indices) > 0:
        component_count += 1
        visited = dfs_4_ways(input_image, white_pixels_indices.pop(), visited)
        white_pixels_indices = white_pixels_indices - visited

    return component_count


def number_of_regions_8_ways(input_image):
    rounded_input_images = mapping_grey_scale_to_01(input_image)

    black_pixels_indices = set(np.nonzero(rounded_input_images)[0])
    white_pixels_indices = set(range(0, 28 * 28)) - black_pixels_indices

    visited = black_pixels_indices
    component_count = 0
    while len(white_pixels_indices) > 0:
        component_count += 1
        visited = dfs_8_ways(input_image, white_pixels_indices.pop(), visited)
        white_pixels_indices = white_pixels_indices - visited

    return component_count


def dfs_4_ways(input_image, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    neighbors = set()
    if (start + 1) % 28 != 0 and input_image[start] == input_image[start + 1]:
        neighbors.add(start + 1)
    if start % 28 != 0 and input_image[start] == input_image[start - 1]:
        neighbors.add(start - 1)
    if start < 756 and input_image[start] == input_image[start + 28]:
        neighbors.add(start + 28)
    if start > 27 and input_image[start] == input_image[start - 28]:
        neighbors.add(start - 28)

    for neighbor in neighbors - visited:
        dfs_4_ways(input_image, neighbor, visited)

    return visited


def dfs_8_ways(input_image, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    neighbors = set()
    if (start + 1) % 28 != 0 and input_image[start] == input_image[start + 1]:  # r
        neighbors.add(start + 1)
    if start % 28 != 0 and input_image[start] == input_image[start - 1]:  # l
        neighbors.add(start - 1)
    if start < 756 and input_image[start] == input_image[start + 28]:  # d
        neighbors.add(start + 28)
    if start > 27 and input_image[start] == input_image[start - 28]:  # u
        neighbors.add(start - 28)
    if start > 27 and (start + 1) % 28 != 0 and input_image[start] == input_image[start - 27]:  # ru
        neighbors.add(start - 27)
    if start > 27 and start % 28 != 0 and input_image[start] == input_image[start - 29]:  # lu
        neighbors.add(start - 29)
    if start < 756 and (start + 1) % 28 != 0 and input_image[start] == input_image[start + 29]:  # rd
        neighbors.add(start + 29)
    if start < 756 and start % 28 != 0 and input_image[start] == input_image[start + 27]:  # ld
        neighbors.add(start + 27)
    for neighbor in neighbors - visited:
        dfs_8_ways(input_image, neighbor, visited)

    return visited


def problem_1(train_images, train_labels_original, test_images, test_labels_original):
    # store weights and biases for each classifier
    weights = []
    biases = []
    for i in range(10):
        target_number = i

        weight, bias = logistic_regression_mini_batch_sgd_loss_ce(train_images, train_labels_original, 100, 1,
                                                                  target_number, 512)
        weights.append(weight)
        biases.append(bias)

        print("accuracy for number ", target_number, ": ",
              calculate_accuracy(weight, bias, train_images, train_labels_original, target_number))

        print("accuracy for number ", target_number, ": ",
              calculate_accuracy(weight, bias, test_images, test_labels_original, target_number))

    print("accuracy: ", ten_digits_accuracy(weights, biases, test_images, test_labels_original) * 100, "%")


def problem_2(train_images, train_labels_original, test_images, test_labels_original):
    # store weights and biases for each classifier
    weights = []
    biases = []
    for i in range(10):
        target_number = i

        weight, bias = logistic_regression_mini_batch_sgd_loss_se(train_images, train_labels_original, 100, 1,
                                                                  target_number, 512)
        weights.append(weight)
        biases.append(bias)

        print("accuracy for number ", target_number, ": ",
              calculate_accuracy(weight, bias, train_images, train_labels_original, target_number))

        print("accuracy for number ", target_number, ": ",
              calculate_accuracy(weight, bias, test_images, test_labels_original, target_number))
        print(train_images.shape)

    print("accuracy: ", ten_digits_accuracy(weights, biases, test_images, test_labels_original) * 100, "%")


def problem_3():
    return


def problem_4(train_images, train_labels_original, test_images, test_labels_original):
    from keras.utils import to_categorical

    train_labels = to_categorical(train_labels_original)
    test_labels = to_categorical(test_labels_original)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 10
    history = network.fit(train_images,
                          train_labels,
                          epochs=epochs,
                          batch_size=128,
                          validation_data=(test_images, test_labels))
    return history





def problem_5(train_images, train_labels_original, test_images, test_labels_original):
    number_of_components_4_ways = np.zeros(5)
    number_of_components_8_ways = np.zeros(5)
    p = Pool(16)
    number_of_components_4_ways = p.map(number_of_regions_4_ways, train_images[0:5])
    print("done 4 way")
    number_of_components_8_ways = p.map(number_of_regions_8_ways, train_images[0:5])
    print("done 8 way")

    number_of_components_4_ways = np.array(number_of_components_4_ways)
    number_of_components_8_ways = np.array(number_of_components_8_ways)

    number_of_components_4_ways = number_of_components_4_ways / np.max(number_of_components_4_ways)
    number_of_components_8_ways = number_of_components_8_ways / np.max(number_of_components_8_ways)
    test_case = train_images[0:5]
    train_images = np.column_stack((test_case, number_of_components_4_ways))
    train_images = np.column_stack((test_case, number_of_components_8_ways))


def mnist_classifier():
    # loading mnist data
    (train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

    # normalizing the input
    train_images = train_images_original.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images_original.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    # problem_5(train_images, train_labels_original, test_images, test_labels_original)

    # problem_1(train_images, train_labels_original, test_images, test_labels_original)

    problem_4(train_images, train_labels_original, test_images, test_labels_original)


mnist_classifier()
