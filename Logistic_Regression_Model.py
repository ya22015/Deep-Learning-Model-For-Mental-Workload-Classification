import scipy.io as io
import numpy as np
import pprint as pp
import pandas as pd
from sklearn import datasets
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import dataset 

from google.colab import drive
import scipy.io

# Mount Google Drive
drive.mount('/content/drive')

# Navigate to directory containing the .mat file
%cd '/content/drive/My Drive/neural-network/'

# Load .mat file using scipy.io.loadmat()
data_q = scipy.io.loadmat('WLDataCW.mat')


x_dataset = data_q["data"]
print(x_dataset)
x_dataset.shape

dataset_x = np.array(x_dataset)
dataset_x_2d = dataset_x.reshape(-1, 360)
dataset_x_2d=dataset_x_2d/255
dataset_x_2d.shape
print(dataset_x_2d)

label = data_q["label"]
print(label)
label.shape

#processing the dataset
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(dataset_x_2d.T, label.T, test_size=0.40, random_state=42)

'''
    Let s = sigmoid fuction
'''
def sigmoid_func(x):
  s = 1 / (1 + np.exp(-x))
  return s

def logistic_regression(X, Y, 𝜂, max_iterations):
    """Logistic regression model."""
    # Initialize weights and bias to zeros
    w = np.zeros((X.shape[1], 1))
    b = np.zeros((1,1))
    
    # Iterate until convergence or max_iterations is reached
    for i in range(max_iterations):
            
        '''
            Forward propagation
        '''
        epsilon = 1e-10
        
        z = np.matmul(X, w) + b
        y_output = sigmoid_func(z)
        y_output_ = np.log(y_output)
        y_output_1 = np.log(1 - y_output)
        loss = -(1/X.shape[0])*np.sum((Y*y_output_) + (1-Y)*y_output_1)
        
        # Backward propagation
        a = (y_output - Y)
        dldw = (1/X.shape[0]) * np.matmul(X.T, a)
        dldb = (1/X.shape[0])*np.sum((y_output - Y))
        
        # Update weights and bias
        w_update = w - 𝜂*dldw
        b_update = b - 𝜂*dldb
        
        # Print the cost every 100 iterations
        if i % 100 == 0:
            print(f'loss after iteration {i}: loss={loss:.4f}')
        
        # Check the stopping criteria
        if loss <= 0.0001:
            print(f'Converged after {i} iterations')
            break
        
        # Update weights and bias
        w = w_update
        b = b_update
        
        
    # Return the trained parameters
    return w, b, loss

w, b, loss = logistic_regression(X_train_set, y_train_set, 𝜂 = 0.0001, max_iterations=1000)


# Define a prediction function
def pred(X):
    z = np.matmul(X, w) + b
    y_output = sigmoid_func(z)
    y_output = np.round(y_output)
    return y_output
        
    

# Make predictions on test set
y_pred_result = pred(X_test_set)

# Calculate the accuracy score
print(f'Accuracy on test set: {(np.sum(y_pred_result == y_test_set)/y_test_set.shape[0])*100:.2f}%')
