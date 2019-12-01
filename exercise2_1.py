import numpy 
import matplotlib.pyplot as plt 
import random
import os

def read_data():
    data_path = os.path.join(os.path.join(os.getcwd(), "dataex1"))
    
    f1 = open(os.path.join(data_path, "iris.data"), "r")
    f2 = open(os.path.join(data_path, "pima-indians-diabetes.data"), "r")
    f1_matrix = []
    f2_matrix = []
    
    if f1.mode == 'r':
        f1_matrix = [[var.strip() for var in item.split(',')] for item in f1.readlines()] 
        print(f1_matrix)
    if f2.mode == 'r':
        f2_matrix = [[var.strip() for var in item.split(',')] for item in f2.readlines()] 
        print(f2_matrix)

    return [f1_matrix, f2_matrix]

def exercise2_1():
    data = read_data()

    return
   
    

exercise2_1()