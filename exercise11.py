import numpy 
import matplotlib.pyplot as plt 
import random
 
def least_squares(X, Y):
    theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X)), Y) 
    return theta

def exercise11(N, test, mu, sigma_square, theta_0):
    #Generate N equidistant value points in the interval [0, N]
    N_points = numpy.arange(0, 2, 2/float(N))
    
    #create X using the N points in the interval [0, N]
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    Y_true = numpy.dot(X, theta_0)

    #find Y by multiplying X with theta_0 and adding gaussian eta
    Y_0 = numpy.dot(X, theta_0) + numpy.random.normal(mu, sigma_square, X.__len__())

    #find the theta using least squares, that is find the regression line using the 20 points and Y
    theta = least_squares(X, Y_0)

    Y_least = numpy.dot(X, theta)

    #create a test set and correspondent Y
    X_test = []
    for i in range(0, test):
        x = random.choice(N_points)
        X_test.append([1, x, x**2, x**3, x**4, x**5])

    #find the Y_test
    Y_0_test = numpy.dot(X_test, theta_0) + numpy.random.normal(mu, sigma_square, X_test.__len__())
    Y_test = numpy.dot(X_test, theta)


    #find mean square error
    mse1 = sum((Y_0 - Y_least)**2)/20
    mse2 = sum((Y_test - Y_0_test)**2)/1000

    print(mse1, mse2)

    plt.plot(N_points, Y_0, 'o', label='noisy training set points', color='grey')
    plt.plot(N_points, Y_0, label='curve fitting the data', color='black')

    plt.plot(N_points, Y_least, label='fitted data least', color='blue')

    plt.plot(N_points, Y_true, 'o', label='true curve points', color='red')
    plt.plot(N_points, Y_true, label='true curve', color='red')

    plt.show()






exercise11(20, 1000, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])






