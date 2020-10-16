import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Custom GNU implementation to predict stock prices
class GRU_StockPrice(nn.Module):

    # Class contructor
    def __init__(self, input_size, hidden_size, num_layers, out_features, device):
        
        # Call constructor of the nn.Modue class
        super(GRU_StockPrice, self).__init__()

        # Parameters of the NN
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers= self.num_layers, batch_first=True)

        # End with a fully connected layer that predict a single 
        self.fc = nn.Linear(in_features = self.hidden_size, out_features = out_features)

    # Feed-forward function
    def forward(self, x):
        
        #batch_size = x.size(0)
        batch_size = x.shape[0]

        # Initiate the hiden layers with a tensor full of zeroes
        # h_ has to be a tensor of of shape (num_layers * num_directions, batch, hidden_size)
        # h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # Apply GRU neural network
        out_gru, hidden_layers = self.gru(x, h_0.to(self.device))

        # Apply the fully connected layer to get the output
        result = self.fc(out_gru[:, -1, :]) 

        return result