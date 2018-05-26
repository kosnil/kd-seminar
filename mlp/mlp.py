import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

# read in data
dataset = pd.read_csv("final_data/complete_data.csv")
dataset.head()

# define input vector
X_train = dataset[[ 'articleCount', 'avgSentiment','stdSentiment',
                    '25quantileSentiment', '50quantileSentiment', '75quantileSentiment',
                    'maxSentiment', 'minSentiment', 'Previous_Day_Return']]

plt.figure()
plt.title('Input-Data')
X_train['Previous_Day_Return'].plot(label='Previous-Day Return')
X_train['avgSentiment'].plot(label='AVG Sentiment')
plt.legend()
plt.show()

# define output vector
Y_train = dataset[['Next_Day_Return']]

plt.figure()
plt.title('Next-Day Return')
Y_train.plot()
plt.show()

# TODO - split data into train and test data
# TODO - verify, one network for all companies? - or each company one network?

# create model
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train.values, Y_train.values, epochs=5, batch_size=32)
