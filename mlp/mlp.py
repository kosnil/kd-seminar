import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# TODO - k-fold cross-validation

##########################################################
###     FIRST SETUP: one model for one company         ###
##########################################################

keras.__version__

### DATA ###

# read in data
dataset = pd.read_csv("final_data/complete_data.csv")
dataset = dataset.drop(columns=['Unnamed: 0'])
dataset.head()

# relabel output, so that we create a classification task
dataset['Next_Day_Return_Sign'] = np.where(dataset['Next_Day_Return'] >= 0, 1, 0)

# define features
features = [ 'articleCount', 'avgSentiment','stdSentiment',
             '25quantileSentiment', '50quantileSentiment', '75quantileSentiment',
             'maxSentiment', 'minSentiment', 'socialScore', 'nbOfDuplicates', 'Previous_Day_Return']

# define target variable
target_var = ['Next_Day_Return_Sign']

# preprocess data
scaler      = MinMaxScaler(feature_range=(0, 1))
rescaled    = scaler.fit_transform(dataset[features])

# split data in training and test data
Y_train                          = dataset[target_var]
X_train, X_test, y_train, y_test = train_test_split(rescaled, Y_train, test_size=0.33, random_state=42)

### MODEL ###
model = Sequential()
model.add(Dense(4, activation='relu', input_dim=11))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# summarize layers
model.summary()

history = model.fit(X_train, y_train,
                 epochs=30,
                 verbose=2,
                 validation_data=(X_test, y_test))

plot_model(model, to_file='mlp/mlp.png', show_shapes=True, show_layer_names=True)

### EVALUATTION ###
score           = model.evaluate(X_test, y_test,verbose=1)

# loss
print("\n%s: %.2f" % (model.metrics_names[0], score[0]))
# accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### PREDICTION ###

ynew = model.predict_classes(X_test)
for i in range(len(X_test)):
	print("X=%s, Predicted=%s" % (X_test[i], ynew[i]))