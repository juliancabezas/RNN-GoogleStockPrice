# Import necessary libraries
import numpy as np

# Calculation of the root mean squared error
def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


# Moving window approach for the nextday models
def moving_window_nextday(data, window_size):

    # Empty list were we will store the result
    data_window = []

    n = len(data)

    # create all possible sequences of length seq_len
    for index in range(0, n - window_size): 
        data_window.append(data[index:(index + window_size + 1),:])

    data_window  = np.array(data_window)

    x_data = data_window[:,:-1,:]
    y_data = data_window[:,-1,:]

    return [x_data, y_data]

# Moving window approach for the intraday model
# Here the Open price is used as a predictor for th3e low, high and close prices of the same day
def moving_window_intraday(data, window_size):

    # Empty list were we will store the result
    data_window = []

    n = len(data)

    # create all possible sequences of length seq_len
    for index in range(0, n - window_size): 
        data_window.append(data[index:(index + window_size + 1),:])

    data_window  = np.array(data_window)

    # Create the x data, considering the open price of the last day (day in which we are predicting)
    x_data_no_open = data_window[:,:-1,1:]
    x_data_open = data_window[:,1:,0,None]
    x_data = np.concatenate([x_data_open,x_data_no_open], axis=2)

    # Select low, high and close prices
    y_data = data_window[:,-1,1:4]

    return [x_data, y_data]