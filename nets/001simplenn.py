# http://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))     # this is sigmoid function


def nonlin(x, deriv = False):
    s_x = sigmoid(x)
    if deriv:
        return s_x*(1 - s_x)        # this is its derivative
    return s_x

# Input data set
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# output results
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation deterministic
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

'''
    To get the uniform distributed results in the certain interval
    say [a, b)

    (b - a) * random_sample() + a

    so in above example the range of the values generated : [-1, 1)
'''
# print (syn0)

# generates continuous uniform distribution over stated interval
# print("random generation")
# print(np.random.random((3, 2)))

for iter in range(10000):
    '''forward Propagation'''
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    '''how much did we miss'''
    l1_error = y - l1

    # multiply how much we missed by the slope of the sigmoid
    # at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("output after training")
print(l1)
