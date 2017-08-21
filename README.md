# Stock Price Prediction of Apple Inc. Using Recurrent Neural Network
OHLC Average Prediction of Apple Inc. Using LSTM Recurrent Neural Network

# Dataset:
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of Apple Inc. stocks from 3rd january 2011 to 13th August 2017 - total 1664 rows. 
# Price Indicator:
Stock traders mainly use three indicators for prediction: OHLC average (average of Open, High, Low and Closing Prices), HLC average (average of High, Low and Closing Prices) and Closing price, In this project, OHLC average has been used.
# Data Pre-processing:
After converting the dataset into OHLC average, it becomes one column data. This has been converted into two column time series data, 1st column consisting stock price of time t, and second column of time t+1. All values have been normalized between 0 and 1.
# Version:
Python 2.7
Latest versions of all libraries including deep learning library Keras and Tensorflow.
# Training:
75% data is used for training. Adagrad (adaptive gradient algorithm) optimizer is used for faster convergence.
After training starts it will look like:


# Test:
Test accuracy metric is root mean square error (RMSE).
# Results:
The comparison of OHLC, HLC and Closing price:

![ttt1](https://user-images.githubusercontent.com/24511419/29501710-76018bbe-864c-11e7-9239-afd8bbf19bb8.png)

After the training the fitted curve with original stock price:

![tt2](https://user-images.githubusercontent.com/24511419/29501783-eb7eccd0-864c-11e7-9c26-0db07dea73c0.png)

