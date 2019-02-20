from keras.datasets import mnist
import numpy as np
from multiprocessing import Pool
from keras import models
from keras import layers


# activation function for problem 1 and problem 2
def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s


# finding the probability of belonging to a class for an input image using the weight and bias
def classify(weight, bias, input_image):
    activation = sigmoid(np.dot(weight.T, input_image.T) + bias)
    return activation


# classify labels to 0 and 1
def binary_classify(weight, bias, input_image):
    predicts = classify(weight, bias, input_image)
    print(predicts.shape)
    for i in range(predicts.shape[1]):
        if predicts.T[i] > 0.5:
            predicts.T[i] = 1
        else:
            predicts.T[i] = 0
    return predicts


# using argmax to output the classifier with the highest probability
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


# training the network using squared error and mini batch sgd
def logistic_regression_mini_batch_sgd_loss_se(train_images, train_labels, epochs, lr, target_label, batch_size):
    modified_train_labels = np.zeros(len(train_labels))
    for i in range(len(train_images)):
        if train_labels[i] == target_label:
            modified_train_labels[i] = 1

    # m is the number of examples
    m = train_images.shape[0]
    # n is the number of features
    n = train_images.shape[1]
    # initializing weight and bias
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


# training the network using binary cross entropy and mini batch sgd
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


# computes the accuracy for each classifier given a specific label
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


# compute the accuracy of using argmax for a set of test images
def ten_digits_accuracy(weights, biases, test_images, test_labels):
    test_predictions = ten_digits_classifier(weights, biases, test_images)
    incorrect_count = np.count_nonzero(test_labels - test_predictions)
    return 1.0 - incorrect_count / len(test_labels)


# activation of the weighted input using softmax
def softmax_classify(weight, bias, input_image):
    activation = softmax(np.dot(weight.T, input_image.T) + bias)
    return activation


# predicts the label of test images after training the training data
def softmax_prediction(weight, bias, input_image):
    predicted_labels = np.zeros((input_image.shape[0]))
    predicted_labels = np.squeeze(np.argmax(softmax_classify(weight, bias, input_image), axis=0))
    return predicted_labels


# calculates the accuracy of predicted labels using softmax
def softmax_accuracy(weight, bias, test_images, test_labels):
    correct_count = 0

    test_predictions = softmax_prediction(weight, bias, test_images)
    for i in range(len(test_labels)):
        if test_predictions[i] == test_labels[i]:
            correct_count += 1
    return correct_count / len(test_labels)


# softmax activation function
def softmax(z):
    z = z - np.max(z, axis=0)
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    return s


# training the network with the input data using softmax and mini batch sgd
def softmax_mini_batch_sgd(train_images, train_labels, epochs, lr, batch_size):
    number_of_examples = train_images.shape[0]
    # n is the number of features
    n = train_images.shape[1]
    # m = 10 is the number of output neurons
    m = 10
    weight = np.zeros((n, m))
    bias = np.zeros((m, 1))
    z = np.zeros(m)
    activations = np.zeros(m)
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(number_of_examples)
        train_images_shuffled = train_images[shuffled_indices]
        train_labels_shuffled = train_labels[shuffled_indices]
        for i in range(0, number_of_examples, batch_size):
            xi = train_images_shuffled[i:i + batch_size]
            yi = train_labels_shuffled[i:i + batch_size]

            # computes weighted input and activation of the weighted input
            z = np.dot(weight.T, xi.T) + bias
            activations = softmax(z)
            activations = np.repeat(activations[np.newaxis], m, axis=0) - np.repeat(np.identity(m)[:, :, np.newaxis],
                                                                                    batch_size, axis=2)
            # derivative of loss function with respect to the weighted input
            modified_dim = yi.T[:, np.newaxis]
            dl_dz = np.repeat(modified_dim, m, axis=1) * activations
            dl_dz = np.sum(dl_dz, axis=0)

            # computes the gradient w.r.t the weights and biases
            gradient_w = 1.0 / batch_size * np.dot(xi.T, dl_dz.T)
            gradient_b = 1.0 / batch_size * np.sum(dl_dz, axis=1, keepdims=True)

            # updating the weights and biases using gradient
            weight = weight - lr * gradient_w
            bias = bias - lr * gradient_b
    return weight, bias


# Round the grey pixels of the images to 1 and 0
def mapping_grey_scale_to_01(input_images):
    return np.ceil(input_images)


# compute the number of white regions using the number of connected components (4 neighbors)
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


# compute the number of white regions using the number of connected components(8 neighbors)
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


# dfs on input images(checking 4 neighbors)
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


# dfs on input images(checking 8 neighbors)
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


# executes problem 1. I chose epochs = 100, learning rate = 1 and batch_size = 512
def problem_1(train_images, train_labels_original, test_images, test_labels_original):
    # store weights and biases for each classifier
    weights = []
    biases = []
    for i in range(10):
        target_number = i

        weight, bias = logistic_regression_mini_batch_sgd_loss_se(train_images, train_labels_original, 100, 1,
                                                                  target_number, 512)
        weights.append(weight)
        biases.append(bias)

        print("training set: accuracy for digit ", target_number, ": ",
              calculate_accuracy(weight, bias, train_images, train_labels_original, target_number))

        print("test set: accuracy for digit ", target_number, ": ",
              calculate_accuracy(weight, bias, test_images, test_labels_original, target_number))

    print("accuracy using argmax: ", ten_digits_accuracy(weights, biases, test_images, test_labels_original) * 100, "%")


# executes problem 2. I chose epochs = 100, learning rate = 1 and batch_size = 512
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

        print("training set: accuracy for digit", target_number, ": ",
              calculate_accuracy(weight, bias, train_images, train_labels_original, target_number))

        print("test set: accuracy for digit ", target_number, ": ",
              calculate_accuracy(weight, bias, test_images, test_labels_original, target_number))

    print("accuracy using argmax: ", ten_digits_accuracy(weights, biases, test_images, test_labels_original) * 100, "%")


# executes problem 3. I chose epochs = 10, learning rate = 0.1 and batch_size = 100
def problem_3(train_images, train_labels_original, test_images, test_labels_original):
    train_labels = (np.arange(np.max(train_labels_original) + 1) == train_labels_original[:, None]).astype(float)
    test_labels = (np.arange(np.max(test_labels_original) + 1) == test_labels_original[:, None]).astype(float)
    weights, bias = softmax_mini_batch_sgd(train_images, train_labels, 10, .1, 100)

    print("predicted labels:", softmax_prediction(weights, bias, train_images))

    print("training set accuracy : ",
          softmax_accuracy(weights, bias, train_images, train_labels_original))

    print("test set accuracy  : ",
          softmax_accuracy(weights, bias, test_images, test_labels_original))


# executes problem 4. I chose epochs = 10, and batch_size = 128 using keras
def problem_4(train_images, train_labels_original, test_images, test_labels_original):
    print("softmax using keras:")
    from keras.utils import to_categorical

    train_labels = to_categorical(train_labels_original)
    test_labels = to_categorical(test_labels_original)

    network = models.Sequential()
    # network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 10
    history = network.fit(train_images,
                          train_labels,
                          epochs=epochs,
                          batch_size=128,
                          validation_data=(test_images, test_labels))
    return history


# adding new features to problem 4
def problem_5(train_images, train_labels_original, test_images, test_labels_original):
    print("improving problem 4 by adding new features:")
    p = Pool(16)
    number_of_components_4_ways_train = p.map(number_of_regions_4_ways, train_images)
    number_of_components_4_ways_test = p.map(number_of_regions_4_ways, test_images)
    print("number of components computed (4 neighbors)")
    number_of_components_8_ways_train = p.map(number_of_regions_8_ways, train_images)
    number_of_components_8_ways_test = p.map(number_of_regions_8_ways, test_images)

    print("number of components computed (8 neighbors)")

    number_of_components_4_ways_train = np.array(number_of_components_4_ways_train)
    number_of_components_4_ways_test = np.array(number_of_components_4_ways_test)

    number_of_components_8_ways_train = np.array(number_of_components_8_ways_train)
    number_of_components_8_ways_test = np.array(number_of_components_8_ways_test)

    number_of_components_4_ways_test = number_of_components_4_ways_test / np.max(number_of_components_4_ways_test)
    number_of_components_4_ways_train = number_of_components_4_ways_train / np.max(number_of_components_4_ways_train)
    number_of_components_8_ways_train = number_of_components_8_ways_train / np.max(number_of_components_8_ways_train)
    number_of_components_8_ways_test = number_of_components_8_ways_test / np.max(number_of_components_8_ways_test)

    number_of_components_4_ways_train = np.reshape(number_of_components_4_ways_train, (len(train_images), 1))
    number_of_components_8_ways_train = np.reshape(number_of_components_8_ways_train, (len(train_images), 1))

    number_of_components_4_ways_test = np.reshape(number_of_components_4_ways_test, (len(test_images), 1))
    number_of_components_8_ways_test = np.reshape(number_of_components_8_ways_test, (len(test_images), 1))

    new_train_images = np.append(train_images, number_of_components_8_ways_train, axis=1)
    new_train_images = np.append(new_train_images, number_of_components_4_ways_train, axis=1)

    new_test_images = np.append(test_images, number_of_components_8_ways_test, axis=1)
    new_test_images = np.append(new_test_images, number_of_components_4_ways_test, axis=1)

    problem_4(new_train_images, train_labels_original, new_test_images, test_labels_original)


# preparing the input and executes all the problems
def mnist_classifier():
    # loading mnist data
    (train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

    # normalizing the input
    train_images = train_images_original.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images_original.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    print("--------Problem 1's output--------:")
    problem_1(train_images, train_labels_original, test_images, test_labels_original)
    print("--------Problem 2's output--------:")
    problem_2(train_images, train_labels_original, test_images, test_labels_original)
    print("--------Problem 3's output--------:")
    problem_3(train_images, train_labels_original, test_images, test_labels_original)
    print("--------Problem 4's output--------:")
    problem_4(train_images, train_labels_original, test_images, test_labels_original)
    print("--------Problem 5's output--------:")
    problem_5(train_images, train_labels_original, test_images, test_labels_original)


mnist_classifier()
