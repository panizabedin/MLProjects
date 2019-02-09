from keras.datasets import mnist
import numpy as np


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s


def classify(weight, bias, input_image):
    activation = sigmoid(np.dot(weight.T, input_image.T) + bias)
    return activation


def binary_classify(weight, bias, input_image):
    predicts = classify(weight, bias, input_image)
    for i in range(predicts.shape[1]):
        if predicts.T[i] > 0.5:
            predicts.T[i] = 1
        else:
            predicts.T[i] = 0
    return predicts


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
            gradient_w = 1.0 / m * np.dot(xi.T, (activation - yi).T)
            gradient_b = 1.0 / m * np.sum(activation - yi)
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
    weight = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        train_images_shuffled = train_images[shuffled_indices]
        train_labels_shuffled = train_labels[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = train_images_shuffled[i:i + batch_size]
            yi = train_labels_shuffled[i:i + batch_size]
            z = np.dot(weight.T, xi.T) + bias
            activation = sigmoid(z)
            gradient_w = 1.0 / m * (((activation - yi) * (activation*(1 - activation))).dot(xi))
            gradient_b = 1.0 / m * (np.sum((activation - yi)*(activation * (1-activation))))
            weight = weight - lr * gradient_w.T
            bias = bias - lr * gradient_b.T

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
    print(correct_count)
    print(len(test_labels))
    return correct_count / len(test_labels)


def new_accuracy(weight, bias, test_images, test_labels, target_label):
    modified_test_labels = np.zeros(len(test_labels))
    for j in range(len(test_labels)):
        if test_labels[j] == target_label:
            modified_test_labels[j] = 1

    test_predictions = binary_classify(weight, bias, test_images)
    test_accuracy = 100 - np.mean(np.abs(test_predictions - modified_test_labels) * 100)
    return test_accuracy


def mnist_classifier():
    # loading mnist data
    (train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

    # normalizing the input
    train_images = train_images_original.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    # modified_train_labels = np.zeros(len(train_labels))

    test_images = test_images_original.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    for i in range(10):
        target_number = i

        weight, bias = logistic_regression_mini_batch_sgd_loss_ce(train_images, train_labels_original, 100, 0.1,
                                                                  target_number, 128)

        print("training set: new accuracy for number ", target_number, ": ",
              new_accuracy(weight, bias, train_images, train_labels_original, target_number))
        print("test set: new accuracy for number ", target_number, ": ",
              new_accuracy(weight, bias, test_images, test_labels_original, target_number))

        # print("accuracy for number ", target_number, ": ",
        #       calculate_accuracy(weight, bias, train_images, train_labels_original, target_number))
        # print("old accuracy for number ", target_number, ": ",
        #       calculate_accuracy(weight, bias, test_images, test_labels_original, target_number))
        # print(train_images.shape)
        # print(train_labels_original.shape)

    # for j in range(1, 1000):
    #     print("original label: ", test_labels_original[j], ", classified as ", target_number, ": ",
    #           "yes" if classify(weight, bias, test_images[j]) > .5 else "no")


mnist_classifier()
