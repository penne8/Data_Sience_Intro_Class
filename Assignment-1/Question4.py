import numpy as np

def main():

    A = np.asarray([[5, 6, 7, 8], [1, 3, 5, 4], [1, 0.5, 4, 2], [3, 4, 3, 1]])
    B = np.asarray([[0.57, 0.56, 0.8, 1], [1.5, 4, 6.7, 4.9], [0.2, 0.1, 1, 0.6], [11, 30, 26, 10]])

    matrix_size = len(B)
    # find least square solution
    D = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        ACoefficients = A[i] @ np.transpose([A[i]])
        BCoefficients = A[i] @ np.transpose([B[i]])
        D[i, i] = BCoefficients / ACoefficients
    print(D)

if __name__ == '__main__':
    main()