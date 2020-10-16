###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 3
# Figures of the Google stock price RNN prediction
####################################

# Import the libraries to do graphs
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set the style of the seaborn graphs
sns.set_style("whitegrid")

#----------------------------------------------------------
# Plot of the four types of price in the data (uncorrected)

# Data reading and creating of timestep field
train = pd.read_csv('./data/Google_Stock_Price_Train.csv', thousands=',')
train['Timestep'] = np.arange(start = 1, stop = len(train) + 1 , step = 1 )

# Melt the data to have all the prices in a single column
train_melt = pd.melt(train[['Timestep','Open','High','Low','Close']],id_vars= ['Timestep'],value_vars=['Open','High','Low','Close'])
train_melt = train_melt.rename(columns = {'variable':'Stock price'})

# Make lineplot with the prices and save it in the latex document folder
ax1 = sns.lineplot(x="Timestep", y="value", hue="Stock price",linewidth=1,data=train_melt).set(ylabel='Stock price ($US)')
plt.savefig('./document_latex/stock_price.pdf')
plt.clf()

#----------------------------------
# Plot of the four types of price in the data (corrected)

# Data Correction
train['Close'] = np.where(train['High'] < train['Close'], train['Close'] / 2.002, train['Close'])

train['Close'] = np.where(train['High'] < train['Close'], train['High'], train['Close'])
train['Close'] = np.where(train['Low'] > train['Close'], train['Low'], train['Close'])

# Melt the data to have all the prices in a single column
train_melt = pd.melt(train[['Timestep','Open','High','Low','Close']],id_vars= ['Timestep'],value_vars=['Open','High','Low','Close'])
train_melt = train_melt.rename(columns = {'variable':'Stock price'})

# Make lineplot with the prices and save it in the latex document folder
ax1 = sns.lineplot(x="Timestep", y="value", hue="Stock price",linewidth=1,data=train_melt).set(ylabel='Stock price ($US)')
plt.savefig('./document_latex/stock_price_corrected.pdf',bbox_inches='tight')
plt.clf()

#-------------------------------------
# Plot of the trading volume variable

# Make lineplot with the volume variable and save it in the latex document folder
ax1 = sns.lineplot(x="Timestep", y="Volume",linewidth=1,data=train)
plt.savefig('./document_latex/prediction_volume.pdf',bbox_inches='tight')
plt.clf()



#--------------------------------------------
# Figures of the predicted vs observed results

# Nextday model

# Data reading
prediction_nextday = pd.read_csv('./results/predicted_values_finalmodel_nextday.csv')
prediction_nextday_novolume = pd.read_csv('./results/predicted_values_finalmodel_nextday_without_volume.csv')
prediction_intraday = pd.read_csv('./results/predicted_values_finalmodel_intraday.csv')

# Create a Timestep field
prediction_nextday['Timestep'] = np.arange(start = 1, stop = len(prediction_nextday)+1, step = 1)
prediction_nextday_novolume['Timestep'] = np.arange(start = 1, stop = len(prediction_nextday_novolume)+1, step = 1 )
prediction_intraday['Timestep'] = np.arange(start = 1, stop = len(prediction_intraday)+1, step = 1 )

# Get the testing data *(last 20 records)
prediction_nextday = prediction_nextday.iloc[(len(prediction_nextday)-20):,:]
prediction_nextday_novolume = prediction_nextday_novolume.iloc[(len(prediction_nextday_novolume)-20):,:]
prediction_intraday = prediction_intraday.iloc[(len(prediction_intraday)-20):,:]


# Melt the dataset to have it suitable for plotting
prediction_nextday_melt = pd.melt(prediction_nextday[['Timestep','Open_pred','High_pred','Low_pred','Close_pred','Volume','Open','High','Low','Close','Volume_pred']],
id_vars= ['Timestep'],value_vars=['Open_pred','High_pred','Low_pred','Close_pred','Volume_pred','Open','High','Low','Close','Volume'])

# Volume prediction plot
pred_vol = prediction_nextday_melt[prediction_nextday_melt.variable.isin(['Volume','Volume_pred'])]
pred_vol = pred_vol.rename(columns = {'variable':'Value'})
pred_vol
pred_vol['Value'] = pred_vol['Value'].replace({'Volume':'Actual'}).replace({'Volume_pred':'Predicted'})
pred_vol = pred_vol.rename(columns = {'value':'Trading Volume'})
ax7 = sns.lineplot(x="Timestep", y="Trading Volume", hue="Value",data=pred_vol)
ax7.set_xticklabels(['','1220','','1225','','1230','','1235',''])
plt.ticklabel_format(style='plain', axis='y')
plt.savefig('./document_latex/prediction_volume.pdf',bbox_inches='tight')
plt.clf()

# Stock price prediction for the nextday model

# Make a grid plot
fig, ax =plt.subplots(1,4)

#Go graph by graph

# Open price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_melt[prediction_nextday_melt.variable.isin(['Open','Open_pred'])],
ax=ax[0]).set(title="Open price prediction",ylabel='Price (US$)')
ax[0].get_legend().remove()

# Low price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_melt[prediction_nextday_melt.variable.isin(['Low','Low_pred'])],
ax=ax[1]).set(title="Low price prediction",ylabel='')
ax[1].get_legend().remove()

# High price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_melt[prediction_nextday_melt.variable.isin(['High','High_pred'])],
ax=ax[2]).set(title="High price prediction",ylabel='')
ax[2].get_legend().remove()

# Close price prediction lineplot, here we will place the legend
pred_close = prediction_nextday_melt[prediction_nextday_melt.variable.isin(['Close','Close_pred'])]
pred_close = pred_close.rename(columns = {'variable':'Value'})
pred_close['Value'] = pred_close['Value'].replace({'Close':'Actual'}).replace({'Close_pred':'Predicted'})
sns.lineplot(x="Timestep", y="value", hue="Value",data=pred_close,
ax=ax[3]).set(title="Close price prediction",ylabel='')

# Set up and print complete plot
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)
fig.savefig('./document_latex/prediction_nextday.pdf',bbox_inches='tight')  
plt.clf()



#-----------------------------------

# Stock price prediction for the nextday model without volume

# Melt the dataset to have it suitable for plotting
prediction_nextday_novolume_melt = pd.melt(prediction_nextday_novolume[['Timestep','Open_pred','High_pred','Low_pred','Close_pred','Open','High','Low','Close']],id_vars= ['Timestep'],value_vars=['Open_pred','High_pred','Low_pred','Close_pred','Open','High','Low','Close'])

# Make a grid plot
fig, ax =plt.subplots(1,4)

#Go graph by graph

# Open price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_novolume_melt[prediction_nextday_novolume_melt.variable.isin(['Open','Open_pred'])],
ax=ax[0]).set(title="Open price prediction",ylabel='Price (US$)')
ax[0].get_legend().remove()

# Low price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_novolume_melt[prediction_nextday_novolume_melt.variable.isin(['Low','Low_pred'])],
ax=ax[1]).set(title="Low price prediction",ylabel='')
ax[1].get_legend().remove()

# High price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_nextday_novolume_melt[prediction_nextday_novolume_melt.variable.isin(['High','High_pred'])],
ax=ax[2]).set(title="High price prediction",ylabel='')
ax[2].get_legend().remove()

# Close price prediction lineplot, here we will place the legend
pred_close = prediction_nextday_novolume_melt[prediction_nextday_novolume_melt.variable.isin(['Close','Close_pred'])]
pred_close = pred_close.rename(columns = {'variable':'Value'})
pred_close['Value'] = pred_close['Value'].replace({'Close':'Actual'}).replace({'Close_pred':'Predicted'})
sns.lineplot(x="Timestep", y="value", hue="Value",data=pred_close,
ax=ax[3]).set(title="Close price prediction",ylabel='')

# Set up and print complete plot
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)
fig.savefig('./document_latex/prediction_nextday_novolume.pdf',bbox_inches='tight')  
plt.clf()


#-----------------------------------

# Stock price prediction for the intraday model

# Melt the dataset to have it suitable for plotting
prediction_intraday_melt = pd.melt(prediction_intraday[['Timestep','High_pred','Low_pred','Close_pred','High','Low','Close']],id_vars= ['Timestep'],value_vars=['High_pred','Low_pred','Close_pred','High','Low','Close'])

# Make a grid plot
fig, ax =plt.subplots(1,3)

#Go graph by graph

# Low price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_intraday_melt[prediction_intraday_melt.variable.isin(['Low','Low_pred'])],
ax=ax[0]).set(title="Low price prediction",ylabel='')
ax[0].get_legend().remove()

# High price prediction lineplot
sns.lineplot(x="Timestep", y="value", hue="variable",data=prediction_intraday_melt[prediction_intraday_melt.variable.isin(['High','High_pred'])],
ax=ax[1]).set(title="High price prediction",ylabel='')
ax[1].get_legend().remove()

# Close price prediction lineplot, here the legend will be placed
pred_close = prediction_intraday_melt[prediction_intraday_melt.variable.isin(['Close','Close_pred'])]
pred_close = pred_close.rename(columns = {'variable':'Value'})
pred_close['Value'] = pred_close['Value'].replace({'Close':'Actual'}).replace({'Close_pred':'Predicted'})
sns.lineplot(x="Timestep", y="value", hue="Value",data=pred_close,
ax=ax[2]).set(title="Close price prediction",ylabel='')

# Set up and print the final plot
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)
fig.savefig('./document_latex/prediction_intraday.pdf',bbox_inches='tight')  
plt.clf()

#------------------------------------------------
# Figure of the loss in the three models

# read the files for the three models
loss_nextday = pd.read_csv('./results/train_loss_nextday.csv')
loss_nextday_wv = pd.read_csv('./results/train_loss_nextday_novolume.csv')
loss_intraday = pd.read_csv('./results/train_loss_intraday.csv')

# Add a column for the model
loss_nextday['Model'] = 'Nextday'
loss_nextday_wv['Model'] = 'Nextday w/o vol.'
loss_intraday['Model'] = 'Intraday'

# Make a single dataframe
loss_full = pd.concat([loss_nextday,loss_nextday_wv,loss_intraday])

# Melt to have all the losses ina single file
loss_full_melt = pd.melt(loss_full,id_vars= ['epoch','Model'],value_vars=['loss','loss_val'])
loss_full_melt = loss_full_melt.rename(columns={'variable':'Dataset'})
loss_full_melt['Dataset'] = loss_full_melt['Dataset'].replace({'loss':'Training'}).replace({'loss_val':'Validation'})


# Make a grid plot with training and validation  loss
fig, ax =plt.subplots(1,3)

#Go graph by graph

# Nextday model loss lineplot
sns.lineplot(x="epoch", y="value", hue="Dataset",data=loss_full_melt[loss_full_melt['Model']=='Nextday'],linewidth = 0.9,alpha=0.7,
ax=ax[0]).set(title="Nextday",ylabel='Mean Square Error (scaled data)',ylim=(0, 0.75))
ax[0].get_legend().remove()

# Nextday without volume model loss lineplot
sns.lineplot(x="epoch", y="value", hue="Dataset",data=loss_full_melt[loss_full_melt['Model']=='Nextday w/o vol.'],linewidth = 0.9,alpha=0.7,
ax=ax[1]).set(title="Nextday w/o Volume",ylabel='',ylim=(0, 0.75))
ax[1].get_legend().remove()

# Intraday model loss lineplot
sns.lineplot(x="epoch", y="value", hue="Dataset",data=loss_full_melt[loss_full_melt['Model']=='Intraday'],linewidth = 0.9,alpha=0.7,
ax=ax[2]).set(title = 'Intraday', ylabel='',ylim=(0, 0.75))

# Set up and print complete plot
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)
fig.savefig('./document_latex/train_val_loss.pdf',bbox_inches='tight')  
