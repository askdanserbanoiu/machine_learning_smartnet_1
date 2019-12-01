import numpy 
import matplotlib.pyplot as plt 
import random
import os

#check page 79-81 for report

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

def least_squares(X, Y):
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X)), Y) 
    return theta

def exercise1_2(N, test, experiments, mu, sigma_square, theta_0):
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create X using the N points in the interval [0, N] 
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    #get true Y
    Y_true = numpy.dot(X, theta_0)

    #create X_2 using the N points in the interval [0, N] 
    X_2 = []
    for i in range(0, 20):
        x = N_points[i]
        X_2.append([1, x, x**2])

    Y_2 = []

    for i in range(0, experiments):
        Y_noisy = numpy.dot(X_2, theta_0[0:3]) + numpy.random.normal(mu, sigma_square, X_2.__len__())
        Y_2.append(numpy.dot(X_2, least_squares(X_2, Y_noisy)))

    Y_avg2 = numpy.average(Y_2, axis=0)
    Y_var2 = numpy.std(Y_2, axis = 0)

    #create X_10 using the N points in the interval [0, N] 
    X_10 = []
    for i in range(0, 20):
        x = N_points[i]
        X_10.append([1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10])
    
    Y_10 = []

    for i in range(0, experiments):
        Y_noisy = numpy.dot(X_10, theta_0 + [0,0,0,0,0]) + numpy.random.normal(mu, sigma_square, X_10.__len__())
        Y_10.append(numpy.dot(X_10, least_squares(X_10, Y_noisy)))

    Y_avg10 = numpy.average(Y_10, axis = 0)
    Y_var10 = numpy.std(Y_10, axis = 0)
    
    print(Y_var10)

    plt.title('Exercise 1_2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(N_points, Y_avg2, label='mean Y 2nd degree pol', color='#000099')
    plt.errorbar(N_points, Y_avg2, yerr=Y_var2, fmt='.k', label='standard deviation from the mean')
    plt.plot(N_points, Y_avg10, label='mean Y 10th degree pol', color='#005800')
    plt.errorbar(N_points, Y_avg10, yerr=Y_var10, fmt='.k')
    plt.plot(N_points, Y_true, 'o', label='true curve points', color='red')
    plt.plot(N_points, Y_true, 'o', color='red')
    plt.legend(bbox_to_anchor=(0.55, 1.0), fontsize='small')
    print_figure("exercise1_2")
    plt.show()


exercise1_2(20, 1000, 100, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])