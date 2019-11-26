import numpy 
import matplotlib.pyplot as plt 
import random
import math
import regression 

def column(matrix, i):
    return [row[i] for row in matrix]

def exercise12(N, test, experiments, mu, sigma_square, theta_0):
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create X using the N points in the interval [0, N] and do regression using 5th degree pol
    X = []
    for i in range(0, N):
        X.append(regression.polynomial_5th_X(N_points[i]))

    #get true Y
    Y_true = regression.polynomial_equation(X, theta_0, 0)

    #create X_2 using the N points in the interval [0, N] and do regression using 2th degree pol for 100 times
    X_2 = []
    for i in range(0, 100):
        X_2.append(regression.polynomial_2nd_X(random.choice(N_points)))

    X_2.sort()

    Y_2 = []

    for i in range(0, experiments):
        Y_i = regression.polynomial_equation(X_2, theta_0[0:3], regression.normal_noise(mu, sigma_square, X_2.__len__()))
        theta_i = regression.least_squares(X_2, Y_i)
        Y_sol = regression.polynomial_equation(X_2, theta_i, 0)
        Y_2.append(Y_sol)

    Y_avg2 = numpy.average(Y_2)
    Y_var2 = numpy.var(Y_2)

    #create X_10 using the N points in the interval [0, N] and do regression using 10th degree pol for 100 times
    X_10 = []
    for i in range(0, experiments):
        X_10.append(regression.polynomial_10th_X(random.choice(N_points)))
    
    X_10.sort()

    Ys_10 = []

    for i in range(0, experiments):
        Y_i = regression.polynomial_equation(X_10, theta_0 + [0,0,0,0,0], regression.normal_noise(mu, sigma_square, X_10.__len__()))
        theta_i = regression.least_squares(X_10, Y_i)
        Y_sol = regression.polynomial_equation(X_10, theta_i, 0)
        Ys_10.append(Y_sol)

    Y_avg10 = numpy.average(Ys_10, axis = 0)
    Y_var10 = numpy.var(Ys_10, axis = 0)


    plt.plot(N_points, Y_true, label='true curve', color='black')
    plt.plot(N_points, Y_true, 'o', label='true curve points', color='grey')
    print(Y_2)
    #plt.plot(column(X_2, 1), Y_avg2, label='average Y 2nd degree pol', color='blue')
    #plt.plot(column(X_2, 1), Y_avg2 + Y_var2, 'o', label='variance of Y 2nd degree pol', color='orange')
    #plt.plot(column(X_2, 1), Y_avg2 - Y_var2, 'o', label='variance of Y 2nd degree pol', color='orange')

    plt.plot(column(X_10, 1), Y_avg10, label='average Y 10th degree pol', color='green')
    plt.plot(column(X_10, 1), Y_avg10 + Y_var10, 'o', label='variance of Y 10th degree pol', color='orange')
    plt.plot(column(X_10, 1), Y_avg10 - Y_var10, 'o', label='variance of Y 10th degree pol', color='orange')


    plt.show()






exercise12(20, 1000, 100, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])