import numpy 
import matplotlib.pyplot as plt 
import random
import math
import os

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return

def ridge_regression(X, Y, l):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(XX_transpose + l*numpy.identity(XX_transpose.__len__())),\
                                numpy.transpose(X)), Y) 
    return theta

def exercise1_3(N, N_test, l_list, mu, sigma_square, theta_0):
    
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**5])

    Y_true = numpy.dot(X, theta_0)

    Y_training = numpy.dot(X, theta_0) + numpy.random.normal(mu, math.sqrt(sigma_square), X.__len__())
    
    X_test = []
        
    for i in range(0, N_test):
        x = random.uniform(0, 2)
        X_test.append([1, x, x**2, x**3, x**5])
        
    Y_training_test = numpy.dot(X_test, theta_0) + numpy.random.normal(mu, math.sqrt(sigma_square), X_test.__len__())

    mse1_all_lamdas = []
    mse2_all_lamdas = []
    
    for i in range(0, l_list.__len__()):

        theta = ridge_regression(X, Y_training, l_list[i])
        Y_ridge_regression = numpy.dot(X, theta)
        Y_test = numpy.dot(X_test, theta)
    
        #find mean square error
        mse1 = sum((Y_training - Y_ridge_regression)**2)/N
        mse2 = sum((Y_training_test - Y_test)**2)/N_test
        
        mse1_all_lamdas.append(mse1)
        mse2_all_lamdas.append(mse2)
    
        #print(mse1)
        #print(mse2)
    
#        plt.title('Exercise 1_3_' + chr(ord('`') + (i + 1)))
#        plt.xlabel('x')
#        plt.ylabel('y')
#        plt.axis([N_points[0], N_points[-1], -0.3, 2])
#        plt.plot(0, label='mse_y='+str(mse2), color='white')
#        plt.plot(N_points, Y_training, label='training set', color='m', marker='*', linestyle='')
#        plt.plot(N_points, Y_ridge_regression, label='ridge regression with l=' + str(+l_list[i]), color='#339900')
#        plt.plot(N_points, Y_true, label='true curve', color='red')
#        plt.plot(N_points, Y_true, 'o', color='red')
#    
#        plt.legend(bbox_to_anchor=(0.53, 1.0), fontsize='small')
#        print_figure("exercise1_3_" + chr(ord('`') + (i + 1)))
#        plt.show()

    plt.title('Exercise 1_3_mse' + chr(ord('`') + (i + 1)))
    plt.xlabel('lamda')
    plt.ylabel('mse1')
    plt.axis([l_list[0], l_list[-1], 0.04, 0.15])
    plt.plot(l_list, mse1_all_lamdas, label='mse over training set', color='m', linestyle='-')
    plt.plot(l_list, mse2_all_lamdas, label='mse over test set', color='g', linestyle='-')
    plt.legend(bbox_to_anchor=(0.53, 1.0), fontsize='small')
    print_figure("exercise1_3" + chr(ord('`') + (i + 1)))
    plt.show()


exercise1_3(20, 1000, numpy.arange(0, 1, 0.02), 0, 0.1, [0.2, -1, 0.9, 0.7, -0.2])