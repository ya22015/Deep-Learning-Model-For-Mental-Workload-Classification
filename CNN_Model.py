import scipy.io as io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

label = data_q["label"]
print(label)
label.shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_dataset.T, label.T, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(32, 3, activation='relu', input_shape=(512, 62, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

 #Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))



# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy:', accuracy)
