import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import binary_crossentropy
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# TODO - split data into train and test data
# TODO - verify, one network for all companies? - or each company one network?

# fix random seed for reproducibility
np.random.seed(7)

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

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

### CREATE MODEL ###

model = Sequential()

# Add layers
model.add(Dense(9, input_dim=9, kernel_initializer='uniform', activation=Activation.relu))
model.add(Dense(1, kernel_initializer='uniform', activation=Activation.sigmoid))

model.compile(loss=binary_crossentropy,
              optimizer=SGD,
              metrics=['accuracy']) # Accuracy performance metric

hist = model.fit(X_train, y_train,
                 epochs=100,
                 batch_size=30,
                 verbose=2,
                 validation_data=(X_test, y_test))

### PREDICTION ###
y_pred = model.predict(X_test)
y_head = np.round(y_pred, 0)

### EVALUATE Classification - MODEL ###

score = model.evaluate(X_test, y_test,verbose=1)

# loss
print("\n%s: %.2f%%" % (model.metrics_names[0], score[0]*100))

# accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

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