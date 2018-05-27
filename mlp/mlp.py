import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


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
dataset['Next_Day_Return_Sign'] = np.where(dataset['Next_Day_Return'] >= 0, 1, 0)
Y_train                         = dataset[['Next_Day_Return_Sign']]

plt.figure()
plt.title('Next-Day Return')
Y_train.plot()
plt.show()

# TODO - split data into train and test data
# TODO - verify, one network for all companies? - or each company one network?

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

### CREATE MODEL ###

model = Sequential()

# Add an input layer
model.add(Dense(X_train.shape[1], activation='relu', input_shape=(X_train.shape[1],)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(y_train.shape[1], activation='sigmoid'))

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, y_train, epochs=200, batch_size=32)

y_pred = model.predict(X_test)

### EVALUATE MODEL ###

score = model.evaluate(X_test, y_test,verbose=1)

print(score)

# Confusion matrix
confusion_matrix(y_test, y_pred)
# Precision
precision_score(y_test, y_pred)
# Recall
recall_score(y_test, y_pred)
# F1 score
f1_score(y_test,y_pred)
# Cohen's kappa
cohen_kappa_score(y_test, y_pred)