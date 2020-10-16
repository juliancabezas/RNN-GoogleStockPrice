
# Assignment 3, Deep Learning Fundamentals, 2020

Julian Cabezas Pena. 
Student ID: a1785086

Testing data augmentation techniques and VGG convolutional neural networks for multiclass image classification on the CIFAR-10 dataset, using Pytorch

## Environment

This repo was tested under a Linux 64 bit OS, using Python 3.7.7 and PyTorch 1.6.0

It was also tested in Google Colab (https://colab.research.google.com/) with GPU enabled

## How to run this repo

In order to use this repo:

1. Clone or download this repo

```bash
git clone https://github.com/juliancabezas/RNN-GoogleStockPrice.git
```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)
3. Create a environment using the environment.yml file included in this repo, using the following command (inside conda or bash)

```bash
conda env create -f environment.yml --name environment
```

4. Activate the conda environment

```bash
conda activate environment
```

5. Run each specific file in yout IDE of preference, (I recommend [VS Code](https://code.visualstudio.com/) with the Python extension), using the root folder of the directory as working directory to make the relative paths work.

It is also possible to run the codes openning a terminal in the project directory
```bash
python <name_of_the_py_file>
```
* Alternatevely, you can use Google Colab (https://colab.research.google.com/) with GPU enabled

Run the codes in order:
- 01-GRU-hyperparameters-nextday.py: Tests different hyperparameters for the GRU neural network that preicts the nextday values of the Google Stock
- 02-GRU-hyperparameters-nextday-without-volume.py: Tests different hyperparameters for the GRU neural network that preicts the nextday values of the Google Stock, exclusing the trading volume
- 03-GRU-hyperparameters-intraDay.py: Tests different hyperparameters for the GRU neural network that preicts the next day (intraday) values of the Google Stock
- 04-GRU-finalmodel-nextday.py: Runs the GRU neural netwoks using the best performing hyperparameters to predict the nextday values of the Google Stock, calculates training and Test RMSE
- 05-GRU-finalmodel-nextday-without-volume.py: Runs the GRU neural netwoks using the best performing hyperparameters to predict the nextday values of the Google Stock (excluding the trading volume), calculates training and Test RMSE
- 06-GRU-finalmodel-intraday.py: Runs the GRU neural netwoks using the best performing hyperparameters to predict the same day (intraday) values of the Google Stock, calculates training and Test RMSE
- 07-Figures.py: Graphs of the stock data, the train/loss curves and predeicted vs observed values for the report (Optional)

Additionaly, the repo contains two python custom modules that are called in the avobementioned codes:
- custom_nn.py: Containg the GRU neural network class used in all the codes
- utils.py: Contain the sliding window method for the nextday and intraday model, used to generate the x and y data. Also contains the RMSE calculation