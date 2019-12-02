import math
import numpy 
import matplotlib.pyplot as plt 
import random
import os

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


def exercise1_1(N, test, mu, sigma_square, theta_0):
    
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create N X training points in the interval [0, N]
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**5])

    Y_true = numpy.dot(X, theta_0)

    #find Y of the training set by multiplying X with theta_0 and adding gaussian eta
    Y_training = numpy.dot(X, theta_0) + numpy.random.normal(mu, math.sqrt(sigma_square), X.__len__())

    #find the theta using the training set
    theta = least_squares(X, Y_training)

    Y_least_squares = numpy.dot(X, theta)

    #create a test set and correspondent Y of 1000 points randomly selected in the interval [0,2]
    X_test = []
    for i in range(0, test):
       x = random.uniform(0, 2)
       X_test.append([1, x, x**2, x**3, x**5])
        
        
    #find the Y_test
    Y_training_test = numpy.dot(X_test, theta_0) + numpy.random.normal(mu, math.sqrt(sigma_square), X_test.__len__())
    Y_test = numpy.dot(X_test, theta)

    #find mean square error
    mse1 = sum((Y_training - Y_least_squares)**2)/N
    mse2 = sum((Y_training_test - Y_test)**2)/test
    
    print(mse1)
    print(mse2)

    plt.title('Exercise 1_1_Training')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([N_points[0], N_points[-1], -0.3, 2])
    plt.plot(0, label='mse_train='+str(mse1), color='white')
    plt.plot(0, label='mse_test='+str(mse2), color='white')
    plt.plot(N_points, Y_training, label='training set', color='m', marker='*', linestyle='')
    #plt.plot(N_points, Y_training, 'o', color='grey')
    plt.plot(N_points, Y_least_squares, label='least squares', color='g', linewidth=3)
    plt.plot(N_points, Y_true, label='true curve', color='k', linestyle='--')
    #plt.plot(N_points, Y_true, 'o', color='red')


    plt.legend(bbox_to_anchor=(0.53, 1.0), fontsize='small')
    print_figure("exercise1_1_Training")
    
    
    plt.show()

exercise1_1(20, 1000, 0, 0.1, [0.2, -1, 0.9, 0.7, -0.2])
