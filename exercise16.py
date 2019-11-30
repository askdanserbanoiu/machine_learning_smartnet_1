import numpy 
import matplotlib.pyplot as plt 
import random

def column(matrix, i):
    return [row[i] for row in matrix]

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

    while(abs(a - temp_a) > convergence and abs(b - temp_b) > convergence):
        
        sigma_0_y = temp_sigma_0_y
        mu_0_y = temp_mu_0_y
        a = temp_a
        b = temp_b
        
        temp_sigma_0_y = numpy.linalg.inv(a*numpy.identity(XX_transpose.__len__()) + b*XX_transpose)
        temp_mu_0_y = b*numpy.dot(numpy.dot(temp_sigma_0_y, numpy.transpose(X)), Y)
        temp_a = theta_0_size/(numpy.linalg.norm(temp_mu_0_y)**2 + numpy.trace(temp_sigma_0_y))
        temp_b = N/(numpy.linalg.norm(Y - numpy.dot(X, temp_mu_0_y))**2 + numpy.trace(numpy.dot(numpy.dot(X, temp_sigma_0_y), numpy.transpose(X))))
    
    return [sigma_0_y, mu_0_y, a, b]


def exercise16(N, sigma2_n, convergence, theta_0):
    
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    Y_true = numpy.dot(X, theta_0)
    
    Y_biased = numpy.dot(X, theta_0) + numpy.random.normal(0, sigma2_n, X.__len__())
    
    result = expectation_maximization(X, Y_biased, theta_0.__len__(), N, convergence)
      
    sigma_0 = result[0]
    mu_0 = result[1]
    a = result[2]
    b = result[3]

    X_test = []
    
    for i in range(0, 20):
        x = random.uniform(0, 2)
        X_test.append([1, x, x**2, x**3, x**4, x**5]) 
        
    X_test.sort()
        
    Y = numpy.dot(X_test, mu_0)
    
    mse = sum((numpy.dot(X_test, theta_0) - Y)**2)/N
    
    print(mse)

    plt.plot(N_points, Y_true, label='true curve', color='red')
    plt.plot(column(X_test, 1), Y, label='mean curve fitting the data', color='grey')

    plt.legend(bbox_to_anchor=(0.42, 1.0), fontsize='small')
    plt.savefig("exercise16_", quality=99)

    plt.show()    
   
   
    

exercise16(500, 0.05, 0.000006, [0.2, -1, 0.9, 0.7, 0, -0.2])


