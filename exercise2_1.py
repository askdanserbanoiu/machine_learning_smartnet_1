import numpy 
import matplotlib.pyplot as plt 
import random
import os
import re


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

def k_nearest_neighbor_classifier(k, test_set, training_set):
    
    distances = []
    for i in range(0, training_set.__len__() - 1):
        distance = 0
        for j in range(0, training_set[i].__len__() - 2):
            #squared euclidean distance as a metric (it could be malahanobis)
            distance = distance + (training_set[i][j] - test_set[j])**2
            
        distances.append([distance, training_set[i][training_set[i].__len__() - 1]])
    
    distances.sort()
    nearest = most_frequent(column(distances[0:k], 1))
    return nearest

def cross_validation_leave_one_out(training_set):
    
    right_guesses = 0
    wrong_guesses = 0

    for i in range(0, training_set.__len__() - 1):
        result = k_nearest_neighbor_classifier(3, training_set[i], [x for j, x in enumerate(training_set) if j != i])
        if (result == training_set[i][training_set[i].__len__() - 1]):
            right_guesses = right_guesses + 1
        else:
            wrong_guesses = wrong_guesses + 1
            
    frequency_right = (right_guesses/training_set.__len__())*100
    frequency_wrong = (wrong_guesses/training_set.__len__())*100
    
    return [frequency_right, frequency_wrong]

    
def exercise2_1():
    data = read_data()
    
    iris = data[0]
    pima = data[1]
    
    classification_percent_iris = cross_validation_leave_one_out(iris)
            
    frequency_right_iris = classification_percent_iris[0]
    frequency_wrong_iris = classification_percent_iris[1]
    
    print(frequency_right_iris)
    print(frequency_wrong_iris)
    
    classification_percent_pima = cross_validation_leave_one_out(pima)

    frequency_right_pima =  classification_percent_pima[0]
    frequency_wrong_pima =  classification_percent_pima[1]
    
    print(frequency_right_pima)
    print(frequency_wrong_pima)

    return
   
    

exercise2_1()