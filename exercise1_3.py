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

def ridge_regression(X, Y, l):
    XX_transpose = numpy.dot(numpy.transpose(X), X)
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(XX_transpose + l*numpy.identity(XX_transpose.__len__())), numpy.transpose(X)), Y) 
    return theta

def exercise1_3(N, N_test, l_list, mu, sigma_square, theta_0):
    
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    Y_true = numpy.dot(X, theta_0)

    Y_training = numpy.dot(X, theta_0) + numpy.random.normal(mu, sigma_square, X.__len__())
    
    X_test = []
        
    for i in range(0, N_test):
        x = random.uniform(0, 2)
        X_test.append([1, x, x**2, x**3, x**4, x**5])
        
    Y_training_test = numpy.dot(X_test, theta_0) + numpy.random.normal(mu, sigma_square, X_test.__len__())

    for i in range(0, l_list.__len__()):

        theta = ridge_regression(X, Y_training, l_list[i])
        Y_ridge_regression = numpy.dot(X, theta)
        Y_test = numpy.dot(X_test, theta)
    
        #find mean square error
        mse1 = sum((Y_training - Y_ridge_regression)**2)/N
        mse2 = sum((Y_training_test - Y_test)**2)/N_test
    
        print(mse1)
        print(mse2)
    
        plt.title('Exercise 1_3_' + chr(ord('`') + (i + 1)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(0, label='mse_y='+str(mse2), color='white')
        plt.plot(N_points, Y_training, label='training set', color='grey')
        plt.plot(N_points, Y_training, 'o', color='grey')
        plt.plot(N_points, Y_ridge_regression, label='prediction ridge regression with l=' + str(+l_list[i]), color='#339900')
        plt.plot(N_points, Y_true, label='true curve', color='red')
        plt.plot(N_points, Y_true, 'o', color='red')
    
        plt.legend(bbox_to_anchor=(0.6, 1.0), fontsize='small')
        print_figure("exercise1_3_" + chr(ord('`') + (i + 1)))
        plt.show()




exercise1_3(20, 1000, [0.000006, 0.1, 0.05], 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])


