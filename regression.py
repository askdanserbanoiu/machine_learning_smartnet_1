import numpy 

def polynomial_2nd_X(x):
    X = [1, x, x**2]
    return X

def polynomial_10th_X(x):
    X = [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10]
    return X

def polynomial_5th_X(x):
    X = [1, x, x**2, x**3, x**4, x**5]
    return X

def polynomial_equation(X, theta, noise = 0):
    ym = numpy.dot(X, theta) + noise
    return ym

def normal_noise(mu, sigma_square, n):
    random = numpy.random.normal(mu, sigma_square, n)
    return random

def least_squares(X, Y):
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X)), Y) 
    return theta

def ridge_regression(X, Y, l):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(XX_transpose + l*numpy.identity(XX_transpose.__len__())), numpy.transpose(X)), Y) 
    return theta

def MSE(theta_test, theta_0):
    mean_squared_error = (theta_test - theta_0)**2
    return mean_squared_error