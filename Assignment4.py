# Jeronimo Martinez Barragan
# CSC 362
# Assignment 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotting function
def plot (a, b):
    x_points = np.array(a)
    y_points = np.array(b)
    plt.plot(x_points,y_points, 'o')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss value")
    plt.show()

# Model: oxycon = b + w1 × age + w2 × heartrate
# Load the data
data = np.loadtxt("dataset.txt", skiprows=1)
# Initialise parameters
age = data[:,2]
heartRate = data[:,3]
oxycon = data[:,1]
w1 = 0.0
w2 = 0.0
b = 0.0
# Normalize
age = age / np.max(np.abs(age))
heartRate = heartRate / np.max(np.abs(heartRate))
oxycon = oxycon / np.max(np.abs(oxycon))
# Hyperparameter
learning_rate = 0.001
# Empty lists for plotting the updates
temp_i = []
temp_SE = []
# Empty dict to store optimal values
dict = {}

# Gradient descent function
def gradient_descent(x1, x2, y, w1, w2, b, learning_rate):

    loss_w1 = 0.0
    loss_w2 = 0.0
    loss_b = 0.0
    N = x1.shape[0]
    M = x2.shape[0]

    # loss = (y-(wx+b))**2 
    # Iteratively update the partial derivative for 
    # the w1, w2 and b
    for i,j,k in zip(x1, x2, y):
        loss_w1 += -2*i*(k-(w1*i+b))
        loss_w2 += -2*j*(k-(w2*j+b))
        loss_b += -2*(k-(w1*i+b))
    
    # Update the values and return them
    w1 = w1 - learning_rate*(1/N)*loss_w1
    w2 = w2 - learning_rate*(1/M)*loss_w2
    b = b - learning_rate*(1/N)*loss_b
    return w1, w2, b

# Iteratively make updates
for i in range(500):
    # Run gradient descent
    w1, w2, b = gradient_descent(age, heartRate, oxycon, w1, w2, b, learning_rate)
    # Local variable for the equation
    y = w1*age + w2*heartRate + b
    # Sum of squared error
    SE = np.divide(np.sum((oxycon-y)**2, axis=0), age.shape[0])
    # Append values to the lists
    temp_i.append(i)
    temp_SE.append(SE)
    # Print and graph every certain number of iterations
    if i%100 == True:
        plot(temp_i, temp_SE)
        dict[SE] = [w1, w2, b]
        print(f'Iteration {i}, loss value is {SE}, parameters weight: {w1}, {w2}, bias: {b}')

# Print the optimal values
opt_loss = min(dict.keys())
opt_w1 = dict[opt_loss][0]
opt_w2 = dict[opt_loss][1]
opt_b = dict[opt_loss][2]
print()
print(f"""Optimal loss value: {opt_loss}
Optimal weights value: {opt_w1} and {opt_w2}
Optimal bias: {opt_b}""")

# Predict oxygen consumption for an astronaut with heart rate of 130 and age 40
# Model: oxycon = b + w1 × age + w2 × heartrate
y = opt_b + (opt_w1*40 + opt_w2*130)
print(f'The oxygen consumption for an astronaut with heart rate of 130 and age 40 is {y}')
