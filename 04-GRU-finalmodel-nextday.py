###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 3
# GRU neural network to predict Google stock prices (Nextday) - Final model (tuned)
####################################

# Importing the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Custom modules
import utils as utils # Moving window and RMSE
import custom_nn as custom_nn # GRU neural network


#----------------------------------------
# Parameters

input_size = 5
out_features = 5
window_size = 40
hidden_size = 30
num_layers = 1
n_epochs = 1500
learning_rate = 0.05


# --------------------------------------

# Data reading
train = pd.read_csv('./data/Google_Stock_Price_Train.csv', thousands=',')
test = pd.read_csv('./data/Google_Stock_Price_Test.csv', thousands=',')

# Divide by 2.002, as thhe split share was from 1000 to 2002
train['Close'] = np.where(train['High'] < train['Close'], train['Close'] / 2.002, train['Close'])

# In case the Close price is placed above High or below Low replace it
train['Close'] = np.where(train['High'] < train['Close'], train['High'], train['Close'])
train['Close'] = np.where(train['Low'] > train['Close'], train['Low'], train['Close'])

# Min max scaling, it will leave the data from 0 to 1

# Open Price
open_min, open_max = train['Open'].min(), train['Open'].max()
train['Open'] = (train['Open'] - open_min) / (open_max - open_min)

# Low Price
low_min, low_max = train['Low'].min(), train['Low'].max()
train['Low'] = (train['Low'] - low_min) / (low_max - low_min)

# High Price
high_min, high_max = train['High'].min(), train['High'].max()
train['High'] = (train['High'] - high_min) / (high_max - high_min)

# Close Price
close_min, close_max = train['Close'].min(), train['Close'].max()
train['Close'] = (train['Close'] - close_min) / (close_max - close_min)

# Volumes
vol_min, vol_max = train['Volume'].min(), train['Volume'].max()
train['Volume'] = (train['Volume'] - vol_min) / (vol_max - vol_min)

# The test data will be rescaled with the same coefficient as the train data, as we are not seeing it yet
test['Open'] = (test['Open'] - open_min) / (open_max - open_min)
test['Low'] = (test['Low'] - low_min) / (low_max - low_min)
test['High'] = (test['High'] - high_min) / (high_max - high_min)
test['Close'] = (test['Close'] - close_min) / (close_max - close_min)
test['Volume'] = (test['Volume'] - vol_min) / (vol_max - vol_min)

# Append the train and test data
train_test = train.append(test)
train_test

# We will get the data using a moving window approach
x_full, y_full = utils.moving_window_nextday(train_test[['Open','High','Low','Close','Volume']].values,window_size)

# Split into train and test data
x_train = x_full[:(len(train) - window_size),:,:]
x_test = x_full[(len(train) - window_size):,:,:]
y_train = y_full[:(len(train) - window_size),:]
y_test = y_full[(len(train) - window_size):,:]

# Convert numpy arrays to tensor
x_train = torch.from_numpy(x_train).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# Set cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Create the GRU neural network
nn_price = custom_nn.GRU_StockPrice(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, out_features = out_features, device = device).to(device)

# use Mean Square Error as loss metric
criterion = torch.nn.MSELoss(reduction='mean')

# Use Adam optimizer (recommended for RNN)
optimiser = torch.optim.Adam(nn_price.parameters(), lr=learning_rate)

print("Training Final model...")

# Go through the epochs trainign the model
for i in range(n_epochs):

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Get the prediction in this epoch
    y_train_pred = nn_price(x_train)

    # Calculate loss
    loss = criterion(y_train_pred, y_train)

    #print("Epoch:", i+1, "Mean Squared Error:", loss.item())

    # Back propagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

print("Training finished")

print("Getting error measures..")


# Get prediction on train and test data
with torch.no_grad():

    # Put the X data in the cpu
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    
    # Predict for validation and test data
    pred_train = nn_price(x_train)
    pred_test = nn_price(x_test)

# Cerate a single vector with all the predicted data
predicted = np.vstack((pred_train.cpu().data.numpy(),pred_test.cpu().data.numpy()))

predicted_df = pd.DataFrame(predicted,columns=['Open_pred','High_pred','Low_pred','Close_pred','Volume_pred'])

# Get the original prices in the predicted dataset
predicted_df['Open_pred'] = (predicted_df['Open_pred'] * (open_max-open_min)) + open_min
predicted_df['High_pred'] = (predicted_df['High_pred'] * (high_max-high_min)) + high_min
predicted_df['Low_pred'] = (predicted_df['Low_pred'] * (low_max-low_min)) + low_min
predicted_df['Close_pred'] = (predicted_df['Close_pred'] * (close_max-close_min)) + close_min
predicted_df['Volume_pred'] = (predicted_df['Volume_pred'] * (vol_max-vol_min)) + vol_min

# Get a dataframe with the true (actual) data in the original scale
actual = train_test[['Open','High','Low','Close','Volume']]
actual = actual.iloc[window_size:,:]
actual['Open'] = (actual['Open'] * (open_max-open_min)) + open_min
actual['High'] = (actual['High'] * (high_max-high_min)) + high_min
actual['Low'] = (actual['Low'] * (low_max-low_min)) + low_min
actual['Close'] = (actual['Close'] * (close_max-close_min)) + close_min
actual['Volume'] = (actual['Volume'] * (vol_max-vol_min)) + vol_min

# Create a pandas dataframe with the predicted and observed and store it in a csv
df_pred = pd.concat([predicted_df.reset_index(), actual.reset_index()], axis=1)
df_pred.to_csv('results/predicted_values_finalmodel_nextday.csv', index = False)

# Print the RMSE of the train data
print('Train RMSE (Open)= ', utils.rmse(predicted_df.iloc[:len(test),0].values,actual.iloc[:len(test),0].values))
print('Train RMSE (Low)= ', utils.rmse(predicted_df.iloc[:len(test),1].values,actual.iloc[:len(test),1].values))
print('Train RMSE (High)= ', utils.rmse(predicted_df.iloc[:len(test),2].values,actual.iloc[:len(test),2].values))
print('Train RMSE (Close)= ', utils.rmse(predicted_df.iloc[:len(test),3].values,actual.iloc[:len(test),3].values))
print('Train RMSE (Volume)= ', utils.rmse(predicted_df.iloc[:len(test),4].values,actual.iloc[:len(test),4].values))

# Print the RMSE of the test data
print('Test RMSE (Open)= ', utils.rmse(predicted_df.iloc[len(test):,0].values,actual.iloc[len(test):,0].values))
print('Test RMSE (Low)= ', utils.rmse(predicted_df.iloc[len(test):,1].values,actual.iloc[len(test):,1].values))
print('Test RMSE (High)= ', utils.rmse(predicted_df.iloc[len(test):,2].values,actual.iloc[len(test):,2].values))
print('Test RMSE (Close)= ', utils.rmse(predicted_df.iloc[len(test):,3].values,actual.iloc[len(test):,3].values))
print('Test RMSE (Volume)= ', utils.rmse(predicted_df.iloc[len(test):,4].values,actual.iloc[len(test):,4].values))







