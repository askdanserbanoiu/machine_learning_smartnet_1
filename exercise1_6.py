import numpy 
import matplotlib.pyplot as plt 
import random
import math
import os

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

def expectation_maximization(X, Y, theta_0_size, N, convergence): 
    sigma_0_y = []
    mu_0_y = []
    a = 1
    b = 1    
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    
    temp_sigma_0_y = []
    temp_mu_0_y = []
    temp_a = 0.5
    temp_b = 0.5
 
    a_values = [1]
    b_values = [1]
    while(abs(a - temp_a) > convergence and abs(b - temp_b) > convergence):
        
        sigma_0_y = temp_sigma_0_y
        mu_0_y = temp_mu_0_y
        a = temp_a
        b = temp_b
        a_values.append(a)
        b_values.append(b)
        
        temp_sigma_0_y = numpy.linalg.inv(a*numpy.identity(XX_transpose.__len__()) + b*XX_transpose)
        temp_mu_0_y = b*numpy.dot(numpy.dot(temp_sigma_0_y, numpy.transpose(X)), Y)
        temp_a = theta_0_size/(numpy.linalg.norm(temp_mu_0_y)**2 + numpy.trace(temp_sigma_0_y))
        temp_b = N/(numpy.linalg.norm(Y - numpy.dot(X, temp_mu_0_y))**2 + numpy.trace(numpy.dot(numpy.dot(X, temp_sigma_0_y), numpy.transpose(X))))
    
    return [sigma_0_y, mu_0_y, a_values, b_values]


def exercise1_6(N, sigma2_n, convergence, theta_0):
    
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**5])

    Y_true = numpy.dot(X, theta_0)
    
    Y_training = numpy.dot(X, theta_0) + numpy.random.normal(0, math.sqrt(sigma2_n), X.__len__())
    
    result = expectation_maximization(X, Y_training, theta_0.__len__(), N, convergence)
      
    sigma_0 = result[0]
    mu_0 = result[1]
    a_values = result[2]
    b_values = result[3]
    

    X_test = []
    
    for i in range(0, 20):
        x = random.uniform(0, 2)
        X_test.append([1, x, x**2, x**3, x**5]) 
        
    X_test.sort()
        
    Y_em = numpy.dot(X_test, mu_0)
    
    mse = sum((numpy.dot(X_test, theta_0) - Y_em)**2)/N
    print(mse)
    std = numpy.std([numpy.dot(X_test, theta_0), Y_em], axis = 0)

    plt.title('Exercise 1_6')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([N_points[0], N_points[-1], -0.3, 2])
    plt.plot(0, label='mse_y='+str(mse), color='white')
    plt.plot(N_points, Y_true, label='true curve', color='red')
    plt.plot(column(X_test, 1), Y_em, label='expectation maximization curve', color='grey')
    plt.errorbar(column(X_test, 1), Y_em, yerr=std, fmt='.k', label='standard deviation')
    plt.legend(bbox_to_anchor=(0.55, 1.0), fontsize='small')
    print_figure("exercise1_6")
    plt.show()    
   
    plt.title('Exercise 1_6_convergence')
    plt.xlabel('Iteration')
    plt.ylabel('sigma')
    plt.axis([0, b_values.__len__() - 1, 0, 2])
    plt.plot(range(0, b_values.__len__()), numpy.true_divide(1, b_values), label='noise_variance', color='g')
    plt.hlines(sigma2_n, 0, b_values.__len__(), colors='g', linestyles='--', label='noise_variance_true')
    plt.legend(bbox_to_anchor=(0.55, 1.0), fontsize='small')
    print_figure("exercise1_6")
    plt.show()    
    

exercise1_6(500, 0.05, 0.0000001, [0.2, -1, 0.9, 0.7, -0.2])