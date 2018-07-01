import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# TODO - use previous day return as additional feature dim

# **Read in financial Data**
fin_data = pd.read_csv("finance_data/data/aggregated_returns.csv", index_col=["Timestamp"], parse_dates=True)
fin_data = fin_data.drop(columns=["Unnamed: 0"])
fin_data.head()

fin_data_class = fin_data.applymap(lambda x: 0 if x < 0 else 1)

# **Read in training Data**
path_to_data = "doc2vec/data/article_vectors_2017-09-28-2018-06-18.json"
data = pd.read_json(path_to_data)
data.head()

data_dict = {}

fin_data_dates = fin_data.index.date.tolist()
data_dates = data.index.date.tolist()

available_dates = list(set(fin_data_dates).intersection(data_dates))
fin_data_class = fin_data_class[fin_data_class.index.isin(available_dates)]
data = data[data.index.isin(available_dates)]
data = data['Apple']
data.shape
len(available_dates)

no_of_articles = 50 # 50 articles per day
no_of_attributes = 100 # 100 doc2vec attributes
no_of_days = len(fin_data_class['Apple'].values) # 148 days

X = np.zeros((no_of_days, no_of_articles, no_of_attributes))
Y = np.zeros((no_of_days, 1))

for date in range(0, no_of_days):
    Y[date] = fin_data_class['Apple'].values[date]
    for article in range(0, no_of_articles):
        X[date][article] = data[date][article]
        Y[date] = fin_data_class['Apple'].values[date]

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=1)

def bidirectional_lstm_model(no_of_articles, no_of_attributes):
    print('Building LSTM model...')
    model = Sequential()
    model.add(LSTM(rnn_size, activation="relu", input_shape=(no_of_articles, no_of_attributes)))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    print('LSTM model built.')

    return model

rnn_size = 100 # size of RNN
vector_dim = 100
learning_rate = 0.0001 #learning rate

model_sequence = bidirectional_lstm_model(no_of_articles, no_of_attributes)

batch_size = 30 # minibatch size
history = model_sequence.fit(x_train, y_train,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=36,
                 validation_split=0.3)

#save the model

save_dir = os.path.join(os.getcwd(), 'mlp')
model_name = 'models/my_model_sequence_lstm.final.hdf5'
model_path = os.path.join(save_dir, model_name)
model_sequence.save(model_path)
print('Saved trained model at %s ' % model_path)

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

ynew = model_sequence.predict_classes(x_val)
for i in range(len(x_val)):
	print("X=%s, Predicted=%s" % (y_val[i], ynew[i]))