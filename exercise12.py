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
    for i in range(0, 20):
        x = N_points[i]
        X_2.append([1, x, x**2])

    Y_2 = []

    for i in range(0, experiments):
        Y_noisy = numpy.dot(X_2, theta_0[0:3]) + numpy.random.normal(mu, sigma_square, X_2.__len__())
        Y_2.append(numpy.dot(X_2, least_squares(X_2, Y_noisy)))

    Y_avg2 = numpy.average(Y_2, axis=0)
    Y_var2 = numpy.var(Y_2, axis=0)

    #create X_10 using the N points in the interval [0, N] and do regression using 10th degree pol for 100 times
    X_10 = []
    for i in range(0, 20):
        x = N_points[i]
        X_10.append([1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10])
    
    Y_10 = []

    for i in range(0, experiments):
        Y_noisy = numpy.dot(X_10, theta_0 + [0,0,0,0,0]) + numpy.random.normal(mu, sigma_square, X_10.__len__())
        Y_10.append(numpy.dot(X_10, least_squares(X_10, Y_noisy)))

    Y_avg10 = numpy.average(Y_10, axis = 0)
    Y_var10 = numpy.var(Y_10, axis = 0)

    
    plt.plot(N_points, Y_2[20], color='#000040')
    plt.plot(N_points, Y_2[40], color='#000060')
    plt.plot(N_points, Y_2[60], color='#000080')
    plt.plot(N_points, Y_2[80], color='#0000A0')
    plt.plot(N_points, Y_2[99], color='#0033FF')

    plt.plot(N_points, Y_avg2, label='average Y 2nd degree pol', color='#000099')
    plt.plot(N_points, Y_avg2, 'o', label='average Y 2nd degree pol points', color='#000099')
    plt.errorbar(N_points, Y_avg2, yerr=Y_var2, fmt='.k')


    plt.plot(N_points, Y_10[20], color='#006800')
    plt.plot(N_points, Y_10[40], color='#008800')
    plt.plot(N_points, Y_10[60], color='#00A000')
    plt.plot(N_points, Y_10[80], color='#00D800')
    plt.plot(N_points, Y_10[99], color='#00FF00')

    plt.plot(N_points, Y_avg10, label='average Y 10th degree pol', color='#005800')
    plt.plot(N_points, Y_avg10, 'o', label='average Y 10th degree pol points', color='#005800')
    plt.errorbar(N_points, Y_avg10, yerr=Y_var10, fmt='.k')

    plt.plot(N_points, Y_true, label='true curve', color='red')
    plt.plot(N_points, Y_true, 'o', label='true curve points', color='red')

 
    plt.legend(bbox_to_anchor=(0.5, 1.0), fontsize='small')
    plt.savefig("exercise12", quality=99)
    plt.show()


exercise12(20, 1000, 100, 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])