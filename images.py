from parse_file import LabelData, ImageData, Image, DataFile
import network as n
import numpy as np
import matplotlib.pyplot as plt
import time

training_images = ImageData('training_images')
training_labels = LabelData('training_labels')

num = 1000
pixels = 28 * 28 * num
image_skip = 4+4+4+4
label_skip = 4 + 4
learning_rate = 1.2


X_ = np.frombuffer(training_images.file.read(), dtype=np.uint8, count=pixels, offset=image_skip)
Y_ = np.frombuffer(training_labels.file.read(), dtype=np.uint8, count=num, offset=label_skip)

Y = np.zeros((num * 10))

for i in range(0, num):
    number = Y_[i]
    index = i * 10 + number
    Y[index] = 1

X = X_ / 255.0

X = X.reshape((num, 28 * 28))
Y = Y.reshape((num, 10))

#print(Image(X[1]).print())

print(X.shape, " X shape")
print(Y.shape, " Y shape")

sizes = n.layer_sizes(X, Y)
print(sizes, " Layer sizes")

parameters = n.initialize_parameters(*sizes)

for i in range(0, 1):
    A2, cache = n.forward(X[i], parameters)

    cost = n.compute_cost(A2, Y[i], parameters)

    #grads = n.backward(parameters, cache, X, Y)

    #parameters = n.update_parameters(parameters, grads, learning_rate)

    #print(A2)
