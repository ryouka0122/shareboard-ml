import sys
import numpy as np


def printf(fmt: str, *args):
    sys.stdout.write(fmt % args)


def read_file(path):
    with open(path) as fd:
        return np.array(
            [
                [float(t) for t in line.split(',')] for line in fd
            ]
        )


def compute_cost(X, y, theta):
    m = y.shape[0]
    J = 0.0

    for i in range(m):
        x = X[i]
        J += (x.dot(theta) - y[i]) ** 2

    return J / (2.0 * m)


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)
    K = alpha / m

    for iter in range(iterations):

        grad = np.zeros(n)
        for i in range(m):
            x = X[i]
            grad += x * (x.dot(theta) - y[i])

        theta -= K * grad

    return theta


def main():
    # == load dataset ==========================================
    dataset = read_file('ex1data1.txt')
    m = dataset.shape[0]
    printf('dataset size: %d\n', m)

    # == extend dataset to variables ==========================================
    X = np.array(
        [
            [1.0, x] for x in dataset[:, 0]
        ]
    )
    y = dataset[:, 1]

    # == Compute cost ==========================================
    printf('\n\n------------------------------------------------------\n')
    printf('theta=[ 0, 0] -> %f\n', compute_cost(X, y, np.array([ 0.0, 0.0])))
    printf('theta=[-1, 2] -> %f\n', compute_cost(X, y, np.array([-1.0, 2.0])))

    # == Gradient Descent ==========================================
    iterations = 1500
    alpha = 0.01

    theta = gradient_descent(X, y, np.array([0.0, 0.0]), alpha, iterations)

    printf('\n\n------------------------------------------------------\n')
    printf('Optimized parameter theta using gradient descent\n')
    printf('theta=[%f, %f]\n', theta[0], theta[1])

    # == Predict ==========================================
    printf('\n\n------------------------------------------------------\n')
    for test in [3.5, 7.0, 10.0, 12.0, 15.0, 20.0]:
        data = np.array([1.0, test])
        result = data.dot(theta)
        printf('For Population = %f, predict a profit of %f\n',
               test*10000, result)


if '__main__' == __name__:
    main()
