import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from keras.utils import plot_model
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout

from keras.layers import LeakyReLU

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# TODO - k-fold cross-validation

##########################################################
###     FIRST SETUP: one model for all companies       ###
##########################################################

seed = 7
keras.__version__
colors = ['b', 'c', 'y', 'm', 'r']

###################
###     DATA    ###
###################

# read in data
#dataset = pd.read_csv("../final_data/complete_data.csv")
dataset = pd.read_csv("final_data/complete_data.csv")

dataset = dataset.drop(columns=['Unnamed: 0'])
classes = dataset.ID.unique()
nClasses = len(dataset.ID.unique())
dataset.head()

# relabel output, so that we create a classification task
dataset['relabeled_returns'] = np.where(dataset['Next_Day_Return'] >= 0, 1, 0)
dataset.head()

# create Input with date x (features for each company: currently 121) dimensions
dataset.shape
nr_days = int(dataset.shape[0] / nClasses)
# amount of features reduced by Timestamp, ID, Next_Day_Return, relabeled_returns)
nr_feats = dataset.shape[1] - 4
print("Days {}".format(nr_days))
print("Features {}".format(nr_feats))

X = []
labels_to_drop = ["Timestamp", "ID", "Next_Day_Return", "relabeled_returns"]
for d in dataset["Timestamp"].value_counts().index:
    X.append(dataset[dataset["Timestamp"] == d].drop(labels_to_drop, axis=1).values.reshape(1,nr_feats*nClasses))
len(X)

X = np.array(X).reshape(int(nr_days), nr_feats*nClasses)
X.shape

df_labels = dataset[["Timestamp","relabeled_returns"]].copy()
Y = df_labels["relabeled_returns"].values.reshape(nr_days, nClasses)

# preprocess data
scaler      = MinMaxScaler(feature_range=(0, 1))
X_rescaled    = scaler.fit_transform(X)

# split data in training and test data
X_train, X_test, y_train, y_test = train_test_split(X_rescaled, Y, test_size=0.33, random_state=seed)

###################
###     MODEL   ###
###################

def baseline_model():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(121,)))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(nClasses, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='mlp.png', show_shapes=True, show_layer_names=True)

    return model

model   = baseline_model()
history = model.fit(X_train, y_train,
                     epochs=30,
                     verbose=2,
                     validation_data=(X_test, y_test))

###################
###  EVALUATION ###
###################

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

###################
###  PREDICTION ###
###################

ynew = model.predict(X_test)
ynew_labeled = np.array(ynew)

np.place(ynew_labeled, ynew_labeled > 0.5, [1])
np.place(ynew_labeled, ynew_labeled <= 0.5, [0])

# Visualize Buy and Sell Signals
for i in range(ynew_labeled.shape[0]):
    for a in range(ynew_labeled.shape[1]):
        if ynew_labeled[i, a] == 1:
            print("BUY: - ", classes[a], " FORECAST: ", ynew_labeled[i, a], "")
        elif ynew_labeled[i, a] == 0:
            print("SELL: - ",  classes[a], " FORECAST: ", ynew_labeled[i, a])

plt.figure()
plt_pdR = plt.scatter(classes, ynew[0], color=colors[0])
plt.axhline(y=0.5)
plt.xlabel('Companies')
plt.ylabel('Sell / Buy')
plt.show()

plt.figure()
plt_pdR = plt.scatter(classes, ynew_labeled[0], color=colors[0])
plt.xlabel('Companies')
plt.ylabel('Sell / Buy')
plt.show()

###################
###  K-FOLD ###
###################

estimator   = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=2)
kfold       = KFold(n_splits=10, shuffle=True, random_state=seed)
results     = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))