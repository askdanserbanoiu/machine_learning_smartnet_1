import numpy 
import matplotlib.pyplot as plt 
import random
import os
import re


def column(matrix, i):
    return [row[i] for row in matrix]

def slice_2d_columns(matrix, i_start, i_end): 
    return [matrix[i][i_start : i_end] for i in range(0, matrix.__len__())]

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
    det_cov1 = numpy.linalg.det(cov1)
    det_cov2 = numpy.linalg.det(cov2)
    constant = -1/2 * multiplication(m1, cov1, m1) + 1/2 * multiplication(m2, cov2, m2) + numpy.log(p1/p2) + 1/2 * numpy.log(det_cov2/det_cov1)
    g_x = quadratic_term + linear_term + constant

    return g_x

def perceptron_algorithm(x, training_set):
    
   
    return 
 
def cross_validation_leave_one_out(training_set):
    
    right_guesses = 0
    wrong_guesses = 0

    for i in range(0, training_set.__len__() - 1):
        result = perceptron_algorithm(training_set[i], [x for j, x in enumerate(training_set) if j != i])
        if (result == training_set[i][training_set[i].__len__() - 1]):
            right_guesses = right_guesses + 1
        else:
            wrong_guesses = wrong_guesses + 1
            
    frequency_right = (right_guesses/training_set.__len__())*100
    frequency_wrong = (wrong_guesses/training_set.__len__())*100
    
    return [frequency_right, frequency_wrong]


def exercise2_4():
    data = read_data()
    iris = data[0]
    pima = data[1]
        
    

   
    

exercise2_4()