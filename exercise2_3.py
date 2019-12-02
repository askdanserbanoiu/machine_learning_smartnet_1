import numpy 
import matplotlib.pyplot as plt 
import random
import os
import re
import itertools

def column(matrix, i):
    return [row[i] for row in matrix]

def most_frequent(List): 
    return max(set(List), key = List.count) 

def read_data():
    
    data_path = os.path.join(os.path.join(os.getcwd(), "exercise2"))
    
    f1 = open(os.path.join(data_path, "iris.data"), "r")
    f2 = open(os.path.join(data_path, "pima-indians-diabetes.data"), "r")
    f1_matrix = []
    f2_matrix = []
    
    if f1.mode == 'r':
        f1_matrix = [[var.strip() for var in item.split(',')] for item in f1.readlines()] 
    if f2.mode == 'r':
        f2_matrix = [[var.strip() for var in item.split(',')] for item in f2.readlines()] 
        
    for i in range(0, f1_matrix.__len__()):
        for j in range(0, f1_matrix[i].__len__()):
            if re.match(r'^-?\d+(?:\.\d+)?$', f1_matrix[i][j]):
                f1_matrix[i][j] = float(f1_matrix[i][j])
                
    for i in range(0, f2_matrix.__len__()):
        for j in range(0, f2_matrix[i].__len__()):
            if re.match(r'^-?\d+(?:\.\d+)?$', f2_matrix[i][j]):
                f2_matrix[i][j] = float(f2_matrix[i][j])
                
    return [f1_matrix, f2_matrix]

def group_by_class(training, n_classes):    
    matrix = []
    
    for i in range(0, n_classes.__len__()):
        matrix.append([])
        for j in range(0, training.__len__()):
            if (training[j][training[j].__len__() - 1] == n_classes[i]):
                matrix[i].append(training[j])
                
    return matrix

def multiplication(v1, M, v2):
    result = numpy.dot(numpy.dot(numpy.transpose(v1), numpy.linalg.inv(M)), v2)
    return result

def discrimination_function(x, m1, m2, cov1, cov2, p1, p2):
    x = x[0 : x.__len__() - 1]
    
    quadratic_term = 1/2 * (multiplication(x, cov2, x) - multiplication(x, cov1, x))
    linear_term = multiplication(m1, cov1, x) - multiplication(m2, cov2, x)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    constant = -1/2 * multiplication(m1, cov1, m1) + 1/2 * multiplication(m2, cov2, m2) + numpy.log(p1/p2) + 1/2 * numpy.log(det_cov2/det_cov1)
    g_x = quadratic_term + linear_term + constant

    return g_x

def naive_bayes_classifier_2classes(x, training_set):
    
    #training
    n_classes = numpy.unique(column(training_set, training_set[0].__len__() - 1))

    classes = group_by_class(training_set, n_classes)
    
    # calculation of prior probabilities
    probabilities = []
    
    for i in range(0, classes.__len__()):        
        probabilities.append((classes[i].__len__())/training_set.__len__())
    
    # Gaussian distribution parameters 
    
    # calculations of means and variances  per columns per class
    means = []
    variances = []

    for i in range(0, classes[0].__len__() - 1): 
        for j in range(0, classes.__len__()):
            means.append(numpy.mean(column(classes[j], i)))
            variances.append(numpy.var(column(classes[j], i), axis = 0))

    covariances = []
    for i in range(0, variances.__len__()):
        covariances.append(variances[i]*numpy.identity(variances[i].__len__()))

    results = []
    combinations_classes = list(itertools.combinations(n_classes, 2))

    for i in range(0, combinations_classes.__len__()):
        # Discrimination Function 
        m1 = means[combinations_classes[i][0]]
        m2 = means[combinations_classes[i][1]]
        cov1 = covariances[combinations_classes[i][0]]
        cov2 = covariances[combinations_classes[i][1]]
        p1 = probabilities[combinations_classes[i][0]]
        p2 = probabilities[combinations_classes[i][1]]

        results.append(discrimination_function(x, m1, m2, cov1, cov2, p1, p2))
        
    #return n_classes[0] if result > 0 else n_classes[1]
    print(results)   
    
def exercise2_3():
    data = read_data()
    iris = data[0]
    pima_indians_diabetes = data[1]
        
    right_guesses_indians = 0
    wrong_guesses_indians = 0
        
    for i in range(0, pima_indians_diabetes.__len__() - 1):
        result = naive_bayes_classifier_2classes(pima_indians_diabetes[i], [x for j, x in enumerate(pima_indians_diabetes) if j != i])
        if (result == pima_indians_diabetes[i][pima_indians_diabetes[i].__len__() - 1]):
            right_guesses_indians = right_guesses_indians + 1
        else:
            wrong_guesses_indians = wrong_guesses_indians + 1
            
    frequency_right_indians = (right_guesses_indians/pima_indians_diabetes.__len__())*100
    frequency_wrong_indians = (wrong_guesses_indians/pima_indians_diabetes.__len__())*100
    
    print(frequency_right_indians)
    print(frequency_wrong_indians)

   
    

exercise2_3()