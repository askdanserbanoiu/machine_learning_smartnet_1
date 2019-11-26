import numpy 
import matplotlib.pyplot as plt 
import random


def column(matrix, i):
    return [row[i] for row in matrix]

def least_squares(X, Y):
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X)), Y) 
    return theta

def exercise12(N, test, experiments, mu, sigma_square, theta_0):
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create X using the N points in the interval [0, N] and do regression using 5th degree pol
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    #get true Y
    Y_true = numpy.dot(X, theta_0)

    #create X_2 using the N points in the interval [0, N] and do regression using 2th degree pol for 100 times
    X_2 = []
    for i in range(0, 100):
        x = random.choice(N_points)
        X_2.append([1, x, x**2])

    X_2.sort()
    Y_2 = []

    for i in range(0, experiments):
        Y_i = numpy.dot(X_2, theta_0[0:3]) + numpy.random.normal(mu, sigma_square, X_2.__len__())
        theta_i = least_squares(X_2, Y_i)
        Y_sol = numpy.dot(X_2, theta_i)
        Y_2.append(Y_sol)

    Y_avg2 = numpy.average(Y_2)
    Y_var2 = numpy.var(Y_2)

    #create X_10 using the N points in the interval [0, N] and do regression using 10th degree pol for 100 times
    X_10 = []
    for i in range(0, experiments):
        x = random.choice(N_points)
        X_10.append([1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10])
    
    X_10.sort()
    Ys_10 = []

    for i in range(0, experiments):
        Y_i = numpy.dot(X_10, theta_0 + [0,0,0,0,0]) + numpy.random.normal(mu, sigma_square, X_10.__len__())
        theta_i = least_squares(X_10, Y_i)
        Y_sol = numpy.dot(X_10, theta_i)
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