from Kmeans import Kmeans
from loadMNIST_py import MnistDataloader
import numpy as np

mnistDataLoader = MnistDataloader(
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte',
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = mnistDataLoader.load_data()

numbers = []

# normilize the data
for i in range(len(x_train)):
    numbers.append(np.array(x_train[i]).flatten())
    for j in range(len(x_train[i])):
        numbers[i][j] /= 255
print("started kmeans")
kmeans = Kmeans(10, numbers)
kmeans.run()
