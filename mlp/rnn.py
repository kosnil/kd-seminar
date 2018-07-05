import pandas as pd
import numpy as np
import collections
import os
import matplotlib.pyplot as plt

import evaluation.classification as evaluation

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc

grid_search = False
save = False
command_line = False

# Google, BMW, BASF, Samsung  Telefonica  Allianz  Total  Bayer  Tesla  Airbus  Apple

###################
###     DATA    ###
###################

# Read in financial Data
if command_line:
    fin_data = pd.read_csv("finance_data/data/aggregated_returns.csv", index_col=["Timestamp"], parse_dates=True)
else:
    fin_data = pd.read_csv("../finance_data/data/aggregated_returns.csv", index_col=["Timestamp"], parse_dates=True)

fin_data = fin_data.drop(columns=["Unnamed: 0"])
fin_data.head()

fin_data_class = fin_data.applymap(lambda x: 0 if x < 0 else 1)

# Read in training Data
if command_line:
    path_to_data = "doc2vec/data/article_vectors_2016-05-09-2018-06-18.json"
else:
    path_to_data = "../doc2vec/data/article_vectors_2016-05-09-2018-06-18.json"

data = pd.read_json(path_to_data)
data.head()

data_dict = {}

fin_data_dates = fin_data.index.date.tolist()
data_dates = data.index.date.tolist()

available_dates = list(set(fin_data_dates).intersection(data_dates))
fin_data_class = fin_data_class[fin_data_class.index.isin(available_dates)]

# predictions to safe afterwards
df_predictions = pd.DataFrame(columns=fin_data_class.columns, index=fin_data_class.index)
df_predictions.columns

data = data[data.index.isin(available_dates)]

for company in fin_data_class.columns:
    # stable training base, same number of 0 and 1
    # if company == "Allianz" or company == "Samsung" or company == "Google" or company == "Tesla":
    if True:
        print("# ", company, " Occurrences of all classes:")
        print(collections.Counter(fin_data_class[company].values))

        values = data[company].copy()
        values.shape

        length = len(available_dates)

        no_of_articles = 50  # 50 articles per day
        no_of_attributes = 100
        # no_of_attributes = 101  # 100 doc2vec attributes + previous day return
        no_of_days = len(fin_data_class[company].values)  # 480

        X = np.zeros((no_of_days, no_of_articles, no_of_attributes))
        Y = np.zeros((no_of_days, 1))

        for date in range(0, no_of_days):
            Y[date] = fin_data_class[company].values[date]
            if type(values[date]) is list:
                print("Number of Articles: ", len(values[date]))
                size = len(values[date])
            else:
                print("No list")
                size = 0

            for article in range(0, size):
                X[date][article] = values[date][article]
                # X[date][article] = data[date][article] + Y[date].tolist()

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.4, random_state=1)


        ###################
        ###     MODEL   ###
        ###################

        def bidirectional_lstm_model(optimizer="rmsprop", dropout_param=0.1):
            print('Building LSTM model...')
            model = Sequential()
            model.add(Bidirectional(
                LSTM(no_of_attributes, return_sequences=True, activation=K.tanh, recurrent_activation=K.relu),
                batch_input_shape=(None, no_of_articles, no_of_attributes)))
            model.add(LSTM(no_of_attributes))
            model.add(Dropout(dropout_param))
            model.add(Dense(1, activation=K.sigmoid))
            model.compile(loss='binary_crossentropy', optimizer=optimizer,
                          metrics=['acc', evaluation.recall, evaluation.f1])
            print('LSTM model built.')

            return model


        #########################
        ###     Grid-Search   ###
        #########################

        if grid_search:
            my_classifier = KerasClassifier(bidirectional_lstm_model, verbose=2)

            epochs = [3, 20, 40, 100]
            optimizer = ["rmsprop"]

            dropout_param = [0.1, 0.2]
            param_grid = dict(epochs=epochs, optimizer=optimizer, dropout_param=dropout_param)

            validator = GridSearchCV(my_classifier, param_grid=param_grid, n_jobs=1)
            validator.fit(x_train, y_train)

            validator.cv_results_.keys()
            validator.best_params_
            validator.best_score_
            validator.best_index_

            model_sequence = bidirectional_lstm_model(optimizer=validator.best_params_['optimizer'],
                                                      dropout_param=validator.best_params_['dropout_param'])

            history = model_sequence.fit(x_train, y_train,
                                         epochs=validator.best_params_['epochs'],
                                         validation_split=0.3)

        ########################
        ###     Best Model   ###
        ########################

        if grid_search == False:
            model_sequence = bidirectional_lstm_model(optimizer="rmsprop", dropout_param=0.1)

            history = model_sequence.fit(x_train, y_train,
                                         batch_size=25,
                                         epochs=40,
                                         verbose=0,
                                         validation_split=0.3)

        print(model_sequence.summary())

        ###################
        ###  EVALUATION ###
        ###################

        # plot history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # plot history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        y_pred = model_sequence.predict_classes(x_val).ravel()
        for i in range(len(x_val)):
            print("X=%s, Predicted=%s" % (y_val[i], y_pred[i]))

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        print(cm)

        # ROC - Curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred)

        # AUC - Curve
        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='LSTM - Classification (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

        ###################
        ###  PREDICTION ###
        ###################

        preds = model_sequence.predict_classes(X).ravel()
        for i in range(len(X)):
            print("X=%s, Predicted=%s" % (Y[i], preds[i]))

        df_predictions[company] = preds
        # shift returns +1 because it is a prediction
        df_predictions[company] = df_predictions[company].shift(1)


# Read in training Data
if command_line:
    df_predictions.to_csv('mlp/predictions/predictions_rnn.csv', sep='\t')
else:
    df_predictions.to_csv('../mlp/predictions/predictions_rnn.csv', sep='\t')


#############
###  Save ###
#############

if save:
    # save the model
    save_dir = os.path.join(os.getcwd(), 'mlp')
    model_name = 'models/my_model_sequence_lstm.final.hdf5'
    model_path = os.path.join(save_dir, model_name)
    model_sequence.save(model_path)
    print('Saved trained model at %s ' % model_path)
