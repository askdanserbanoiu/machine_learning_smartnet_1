import numpy 
import matplotlib.pyplot as plt 
import random

#check page 591 for report

def column(matrix, i):
    return [row[i] for row in matrix]

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return

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


def exercise1_4(N, sigma2_0, sigma2_n_list, theta_0):
    
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X_true = []
    for i in range(0, N):
        x = N_points[i]
        X_true.append([1, x, x**2, x**3, x**4, x**5])
        
    Y_true = numpy.dot(X_true, theta_0)
    
    X = []
    for i in range(0, N):
        x = random.uniform(0, 2)
        X.append([1, x, x**2, x**3, x**4, x**5])
    X.sort()
    
    for i in range(0, sigma2_n_list.__len__()):
        
        sigma2_n = sigma2_n_list[i]

        Y = numpy.dot(X, theta_0) + numpy.random.normal(0, sigma2_n, X.__len__())
         
        mean_theta = bayesian_inference_mean_theta_y(X, Y, sigma2_0, sigma2_n, theta_0)
        
        mean_y = bayesian_inference_mean_y(X, mean_theta)
        variance_y = bayesian_inference_variance_y(X, sigma2_0, sigma2_n)
            
        plt.plot(N_points, Y_true, label='true curve', color='red')
                
        plt.plot(column(X, 1), mean_y, label='mean curve fitting the data', color='grey')
        plt.errorbar(column(X, 1), mean_y, yerr=variance_y, fmt='.k')
        
        plt.legend(bbox_to_anchor=(0.42, 1.0), fontsize='small')
    
        print_figure("exercise1_4_" + chr(ord('`') + (i + 1)))

        plt.show()
    

exercise1_4(20, 0.1, [0.05, 0.15], [0.2, -1, 0.9, 0.7, 0, -0.2])


