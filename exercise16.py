import numpy 
import matplotlib.pyplot as plt 
import random

#check page 591 for report

def column(matrix, i):
    return [row[i] for row in matrix]

def covariance_sigma(X, sigma2_0, sigma2_n):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    covariance_sigma = numpy.linalg.inv((1/sigma2_0)*numpy.identity(XX_transpose.__len__()) + (1/sigma2_n)*XX_transpose)
    return covariance_sigma

def bayesian_inference_mean_theta_y(X, Y, sigma2_0, sigma2_n, theta_0):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    mean = theta_0 + (1/sigma2_n)*numpy.dot(numpy.dot(covariance_sigma(X, sigma2_0, sigma2_n), numpy.transpose(X)), Y - numpy.dot(X, theta_0))
    return mean

def bayesian_inference_mean_y(X, mean_theta):
    mean = []
    for i in range(0, X.__len__()):
        res = numpy.dot(numpy.transpose(X[i]), mean_theta)
        mean.append(res)       
    return mean

def bayesian_inference_variance_y(X, sigma2_0, sigma2_n):
    variance = []
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    for i in range(0, X.__len__()):
        term = numpy.linalg.inv(sigma2_n*numpy.identity(XX_transpose.__len__()) + sigma2_0*XX_transpose)
        res = sigma2_n + sigma2_n*sigma2_0*sigma2_n*numpy.dot(numpy.dot(numpy.transpose(X[i]), term), X[i])
        variance.append(res)       
    return variance


def exercise16(N, sigma2_0, sigma2_n_list, theta_0):
    return ""
    
   
    

exercise16(20, 0.1, [0.05, 0.15], [0.2, -1, 0.9, 0.7, 0, -0.2])


