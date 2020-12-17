import csv
import matplotlib.pyplot as plt
import numpy as np
global error 
error = [[],[],[]]

whole_data = []
row_data = []
with open('example1.txt') as csv_file:
    # read csv file
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        amount_of_rows = len(row)
        for i in range(len(row)):
            row_data.append(row[i])
        whole_data.append(row_data)
        error[0].append(row_data[-1])       
        row_data = []  
with open('example2.txt') as csv_file:
    # read csv file
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        amount_of_rows = len(row)
        for i in range(len(row)):
            row_data.append(row[i])
        whole_data.append(row_data)
        error[1].append(row_data[-1])       
        row_data = []    
with open('example3.txt') as csv_file:
    # read csv file
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        amount_of_rows = len(row)
        for i in range(len(row)):
            row_data.append(row[i])
        whole_data.append(row_data)
        error[2].append(row_data[-1])       
        row_data = []  
print(error)           
f = open('example4.txt', "a")
for i in range(len(error[0])):
    f.write(f'{i};{str(error[0][i])[1:-1]};{str(error[1][i])[1:-1]};{str(error[2][i])[1:-1]}\n')