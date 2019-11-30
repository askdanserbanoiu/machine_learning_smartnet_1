import numpy 
import matplotlib.pyplot as plt 
import random

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

def exercise1_3(N, test, L_list, mu, sigma_square, theta_0):
    N_points = numpy.arange(0, 2, 2/float(N))
    
    X = []
    for i in range(0, N):
        x = N_points[i]
        X.append([1, x, x**2, x**3, x**4, x**5])

    Y_true = numpy.dot(X, theta_0)

    Y_0 = numpy.dot(X, theta_0) + numpy.random.normal(mu, sigma_square, X.__len__())

    for i in range(0, L_list.__len__()):

        theta = ridge_regression(X, Y_0, L_list[i])
       
        Y_ridge = numpy.dot(X, theta)
       
        X_test = []
        
        for m in range(0, test):
            x = random.choice(N_points)
            X_test.append([1, x, x**2, x**3, x**4, x**5])
    
        Y_0_test = numpy.dot(X_test, theta_0) + numpy.random.normal(mu, sigma_square, X_test.__len__())
    
        Y_test = numpy.dot(X_test, theta)
        
    
        #find mean square error
        mse1 = sum((Y_0 - Y_ridge)**2)/20
        mse2 = sum((Y_test - Y_0_test)**2)/1000
    
        print(mse1)
        print(mse2)
    
        plt.title('Exercise 1_3_' + chr(ord('`') + (i + 1)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(N_points, Y_0, label='noisy curve fitting the data', color='grey')
        plt.plot(N_points, Y_0, 'o', color='grey')
        plt.plot(N_points, Y_ridge, label='ridge regression with l=' + str(+L_list[i]), color='#339900')
        plt.plot(N_points, Y_true, label='true curve', color='red')
        plt.plot(N_points, Y_true, 'o', color='red')
    
        plt.legend(bbox_to_anchor=(0.5, 1.0), fontsize='small')
        print_figure("exercise1_3_" + chr(ord('`') + (i + 1)))
        plt.show()




exercise1_3(20, 1000, [0.2,0.005, 0.5], 0, 0.1, [0.2, -1, 0.9, 0.7, 0, -0.2])


