import sys


def printf(fmt: str, *args):
    sys.stdout.write(fmt % args)


def read_file(path):
    with open(path) as fd:
        return [parse_line(line) for line in fd]


def parse_line(line):
    return [float(t) for t in line.split(',')]


def compute_cost(X, y, theta):
    m = len(y)
    n = len(theta)
    J = 0.0

    for i in range(m):
        x = X[i]
        J += (sum([x[j] * theta[j] for j in range(n)]) - y[i]) ** 2

    return J / (2.0 * m)


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)
    K = alpha / m

    for iter in range(iterations):

        grad = [0.0] * n

        for i in range(m):
            x = X[i]
            cost = sum([x[j] * theta[j] for j in range(n)]) - y[i]

            for j in range(n):
                grad[j] += x[j] * cost

        for j in range(n):
            theta[j] -= K * grad[j]

    return theta


def main():
    # == load dataset ==========================================
    dataset = read_file('ex1data1.txt')
    m = len(dataset)
    printf('dataset size: %d\n', m)

    # == extend dataset to variables ==========================================
    X = []
    y = []
    for data in dataset:
        X.append([1.0, data[0]])
        y.append(data[1])

    # == Compute cost ==========================================
    printf('\n\n------------------------------------------------------\n')
    printf('theta=[ 0, 0] -> %f\n', compute_cost(X, y, [ 0.0, 0.0]))
    printf('theta=[-1, 2] -> %f\n', compute_cost(X, y, [-1.0, 2.0]))

    # == Gradient Descent ==========================================
    iterations = 1500
    alpha = 0.01

    theta = gradient_descent(X, y, [0.0, 0.0], alpha, iterations)

    printf('\n\n------------------------------------------------------\n')
    printf('Optimized parameter theta using gradient descent\n')
    printf('theta=[%f, %f]\n', theta[0], theta[1])

    # == Predict ==========================================
    n = len(theta)
    printf('\n\n------------------------------------------------------\n')
    for test in [3.5, 7.0, 10.0, 12.0, 15.0, 20.0]:
        data = [1.0, test]
        result = sum([data[j] * theta[j] for j in range(n)])
        printf('For Population = %f, predict a profit of %f\n',
               test*10000, result)


if '__main__' == __name__:
    main()
