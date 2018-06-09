import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation

import os
wd = os.getcwd()
print(wd)

df = pd.read_csv("complete_data_0906.csv")
df.head()

# dims = X_train.shape[1]
# print(dims, 'dims')
# print("Building model...")
#
# nb_classes = Y_train.shape[1]
# print(nb_classes, 'classes')
#
# model = Sequential()
# model.add(Dense(nb_classes, input_shape=(dims,), activation='sigmoid'))
# model.add(Activation('softmax'))
#
# model.compile(optimizer='sgd', loss='categorical_crossentropy')
# model.fit(X_train, Y_train)