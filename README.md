This is a Linear Regression model that predicts the amount of oxygen that an astronaut consumes when performing five minutes of intense physical work. The descriptive features for the model will be the age of the astronaut and their average heart rate throughout the work. The regression model is:

oxycon = b + x1 × age + x2 × heartrate

The dataset file shows the data collected for this task.

The python program does the following:

- Loads the dataset in a NumPy array.
- Initializes the model parameters arbitrarily.
- Uses gradient descent to update the model parameters iteratively until convergence. Using batch gradient descent to update the parameters simultaneously for all the samples in the training set.
- Plots the cost function values over the iterations to check the convergence of the model.
- Evaluates the trained model and calculates the sum of squared error between the predicted and actual amount of oxygen that an astronaut will consume.

The program outputs the following:

- Final model parameters over iterations (weights and bias)
- Plot of the cost function values over iterations.
- Sum of squared error.
- The oxygen consumption for an astronaut with heart rate of 130 and age of 40.
