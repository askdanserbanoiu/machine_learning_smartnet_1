import numpy 
import matplotlib 
import random

def bayesian_inference(X, y, theta_0, sigma2_n, sigma2_0):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    mu_a = numpy.linalg.inv(numpy.dot(1/sigma2_0, numpy.identity(XX_transpose.__len__())) + numpy.dot(1/sigma2_n, XX_transpose))
    mu_b = numpy.dot(numpy.linalg.inv(X), y - numpy.dot(X, theta_0))
    mu = theta_0 + numpy.dot(numpy.dot(1/sigma2_n, mu_a), mu_b)


def exercise14(N, test, mu, sigma_square, theta_0):
    
   



exercise14(20, 1000, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])


