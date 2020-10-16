###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 3
# GRU neural network to predict Google stock prices (Intraday) - Hyperparameter tuning
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
out_features = 3

# ---------------------------------------

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


# Set cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ----------------------
# Hyperparameter tuning

# Hyperparameters to be tuned
window_size_array = [20,30,40]
hidden_size_array = [20,30,40]
num_layers_array = [1,2,3]
n_epochs_array = [500,1000,1500]
learning_rate_array = [0.01, 0.05, 0.001]

# Store the results in lists
window_size_list = []
hidden_size_list = []
num_layers_list = []
n_epochs_list = []
learning_rate_list = []
mse_list = []


# Check if the tuning was already done
if not os.path.exists('results/hyperparameter_tuning_intraday.csv'):

    # test different window sizes
    for window_size in window_size_array:

        # We will get the data using a moving window approach
        x_full, y_full = utils.moving_window_intraday(train_test[['Open','High','Low','Close','Volume']].values,window_size)


        # Split into train, validation and test data, the validation data will be the same length as the test set (20)
        x_train = x_full[:(len(train) - window_size - len(test)),:,:]
        x_val = x_full[(len(train) - window_size - len(test)):(len(train) - window_size),:,:]
        x_test = x_full[(len(train) - window_size):,:,:]
        y_train = y_full[:(len(train) - window_size - len(test)),:]
        y_val = y_full[(len(train) - window_size - len(test)):(len(train) - window_size),:]
        y_test = y_full[(len(train) - window_size):,:]

        # Convert numpy arrays to tensor
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        x_val = torch.from_numpy(x_val).type(torch.Tensor)
        y_val = torch.from_numpy(y_val).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        # Test the different hyperparameters
        for hidden_size in hidden_size_array:
            for num_layers in num_layers_array:
                for n_epochs in n_epochs_array:
                    for learning_rate in learning_rate_array:
                        
                        print('Testing hyperparameters:', 'window_size =', window_size, 'hidden_size =', hidden_size,'num_layers =',num_layers,'n_epochs =',n_epochs,'learning_rate =',learning_rate)

                        # Create the GRU neural network
                        nn_price = custom_nn.GRU_StockPrice(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, out_features = out_features, device = device).to(device)

                        # use Mean Square Error as loss metric
                        criterion = torch.nn.MSELoss(reduction='mean')

                        # Use Adam optimizer (recommended for RNN)
                        optimiser = torch.optim.Adam(nn_price.parameters(), lr=learning_rate)

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

                        # Calculate the validation MSE
                        with torch.no_grad():
                            x_val = x_val.to(device)
                            y_val = y_val.to(device)
                            pred_val = nn_price(x_val)
                            loss_val = criterion(pred_val, y_val)

                        # Store the partial results in lists
                        window_size_list.append(window_size)
                        hidden_size_list.append(hidden_size)
                        num_layers_list.append(num_layers)
                        n_epochs_list.append(n_epochs)
                        learning_rate_list.append(learning_rate)
                        mse_list.append(loss_val.item())
                        print('MSE=', loss_val.item())
                        

    # Generate and save a dataframe with all the results as csv
    dic = {'window_size':window_size_list,'hidden_size':hidden_size_list,'num_layers':num_layers_list,'n_epochs':n_epochs_list,'learning_rate':learning_rate_list,'mse_val':mse_list}
    df_tuning = pd.DataFrame(dic)
    df_tuning.to_csv('results/hyperparameter_tuning_intraday.csv', index = False)
    print("Hyperparameter tuning finished")

else:
    # In case the tuning was already performed, read the csv from the results folder
    df_tuning= pd.read_csv('results/hyperparameter_tuning_intraday.csv')
    print("Previous tuning detected")


# Get the row with least error
row_max = df_tuning['mse_val'].argmin()

print('The best combination of hyperparameters is:')
print(df_tuning.iloc[row_max,:])

#--------------------------------------------
# Train the network once again with the best hyperparameters to get the loss curves

# Parameters
window_size = 40
hidden_size = 40
num_layers = 2
n_epochs = 1500
learning_rate = 0.01

# We will get the data using a moving window approach
x_full, y_full = utils.moving_window_intraday(train_test[['Open','High','Low','Close','Volume']].values,window_size)

x_full.shape
y_full.shape

# Split into train, validation adn test data, the validation data will be the same length as the test set (20)
x_train = x_full[:(len(train) - window_size - len(test)),:,:]
x_val = x_full[(len(train) - window_size - len(test)):(len(train) - window_size),:,:]
x_test = x_full[(len(train) - window_size):,:,:]
y_train = y_full[:(len(train) - window_size - len(test)),:]
y_val = y_full[(len(train) - window_size - len(test)):(len(train) - window_size),:]
y_test = y_full[(len(train) - window_size):,:]

# Convert numpy arrays to tensor
x_train = torch.from_numpy(x_train).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
x_val = torch.from_numpy(x_val).type(torch.Tensor)
y_val = torch.from_numpy(y_val).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# Create the GRU neural network again with the new parameters
nn_price = custom_nn.GRU_StockPrice(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, out_features = out_features, device = device).to(device)

# use Mean Square Error as loss metric
criterion = torch.nn.MSELoss(reduction='mean')

# Use Adam optimizer (recommended for RNN)
optimiser = torch.optim.Adam(nn_price.parameters(), lr=learning_rate)

loss_list = []
loss_val_list = []
epoch_list = []

print("Getting the loss curves...")

# Go through the epochs trainign the model
for i in range(n_epochs):

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Get the prediction in this epoch
    y_train_pred = nn_price(x_train)

    # Calculate loss
    loss = criterion(y_train_pred, y_train)

    #print("Epoch:", i+1, "Mean Squared Error:", loss.item())

    # Calculate the validation loss
    with torch.no_grad():
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        pred_val = nn_price(x_val)
        loss_val = criterion(pred_val, y_val)
        
    # Store the loss and epoch in a list
    loss_list.append(loss.item())
    loss_val_list.append(loss_val.item())
    epoch_list.append(i+1)

    # Back propagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# Save the loss curves in a csv 
dic = {'epoch':epoch_list,'loss':loss_list,'loss_val':loss_val_list}
df_train = pd.DataFrame(dic)
df_train.to_csv('results/train_loss_intraday' + '.csv', index = False)
print("Finished!")





