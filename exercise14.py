import numpy 
import matplotlib 
import random
import math
import regression

#   MSE = E[(theta - theta_0)**2]
#   In general E[X] = x_1*p_1 + x_2*p_2 + ... + x_n*p_n which is the average for uniform distribution
#   How to calculate MSE: for every possible value of theta for every training set, subtract it 
#   from theta_0, square it, find the mean for all these theta's
        

def exercise14(N, test, mu, sigma_square, theta_0):
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create X using the N points in the interval [0, N]
    X = []
    for i in range(0, N):
        X.append(regression.polynomial_5th_X(N_points[i]))

    #find Y by multiplying X with theta_0 and adding gaussian eta
    Y_0 = regression.polynomial_equation(X, theta_0, regression.normal_noise(mu, sigma_square, X.__len__()))

    #find the theta using least squares, that is find the regression line using the 20 points and Y
    theta = regression.least_squares(X, Y_0)

    Y_regression = regression.polynomial_equation(X, theta, 0)

    #find mean square error
    mse1 = sum((Y_0 - Y_regression)**2)/20

    #create a test set 
    X_test = []
    for i in range(0, test):
        X_test.append(regression.polynomial_5th_X(random.choice(N_points)))

    #find Y_test
    Y_test = regression.polynomial_equation(X_test, theta, 0)
    Y_0_test = regression.polynomial_equation(X_test, theta_0, regression.normal_noise(mu, sigma_square, X_test.__len__()))


    #find mean square error
    mse2 = sum((Y_test - Y_0_test)**2)/1000

    print(mse1, mse2)




exercise11(20, 1000, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])


