# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:38 2021

@author: anindita
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Concatenate, LSTM, Dropout
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
import tensorflow as tf
import time


def load_path_data(dataset_path, extension):
    path_filenames = list(Path(dataset_path).glob("**/*" + extension))
    return path_filenames


def obtain_feature_data(path_filenames, train=True):
    col = 336 if train == True else 337
    feature_data = np.empty((0, col))
    for file_csv in path_filenames:
        df = read_csv(file_csv)
        temp = df.to_numpy()
        feature_data = np.vstack((feature_data, temp))
    return feature_data


def split_train_val_test(x, y, val_size=0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test


def lstm():
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(1, 5)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=3, activation='relu'))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01)

    regressor.compile(optimizer=adam, loss='mean_squared_error',
                      metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError()])

    regressor.summary()
    plot_model(regressor)
    return regressor


def select_best_feature(feature_data, train=True):
    signal_location = np.arange(6, 306, 3)
    signal = feature_data[:, signal_location]
    max_signal = np.argmax(signal, axis=1)
    feature_max_signal = np.empty((0, 3))
    for idx, row in enumerate(feature_data):
        feature_max_signal = np.vstack(
            (feature_max_signal, row[signal_location[max_signal[idx]] - 1:signal_location[max_signal[idx]] + 2]))
    feature_max_signal = np.hstack((feature_data[:, 0:5], feature_max_signal))
    # feature_clear = feature_max_signal[feature_max_signal[:,7]>-999]
    return feature_max_signal


def collect_bssdi():
    bssdi_location = np.arange(5, 305, 3)
    bssdi = []
    for b in bssdi_location:
        bssdi.append(feature_data[:, b])
    bssdi = np.unique(bssdi)
    return bssdi


def split_x_y(feature_data):
    x = feature_data[:, [0, 2, 5, 6, 7]]
    y = feature_data[:, [1, 3, 4]]
    return x, y


def encode_feature(feature, scaler=None):
    if scaler == None:
        scaler = OrdinalEncoder()
    feature = feature.reshape(-1, 1)
    scaler.fit(feature)
    feature = scaler.transform(feature)
    feature = feature.squeeze()
    return scaler, feature.astype(float)


def normalize_x(x, max_values=None, train=True):
    if train == True:
        max_values = []
        for col in range(x.shape[1]):
            max_values.append(np.max(np.absolute(x[:, col])))
            x[:, col] = x[:, col] / max_values[col]
    else:
        for col in range(x.shape[1]):
            x[:, col] = x[:, col] / max_values[col]
    return max_values, np.absolute(x)


def train_combined():
    Y = y.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.1)

    x_train = x_train.reshape(-1, 1, 5)
    x_test = x_test.reshape(-1, 1, 5)
    y_train = y_train.reshape(-1, 1, 3)
    y_test = y_test.reshape(-1, 1, 3)
    # model = build_model()
    model = lstm()
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stop])

    print('evaluate')
    model.evaluate(x_test, y_test)

    # summarize history for rmse
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['val acc', 'acc'], loc='upper left')
    plt.show()

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('loc rmse')
    plt.ylabel('loss')
    plt.xlabel('val loss rmse')
    plt.legend(['train loc', 'val loc'], loc='upper left')
    plt.show()

    return model


def export_submission(y_pred):
    first_col = []
    for tm in feature_data_test_ori[:, [2, 4, 1]]:
        temp = f'{tm[2]:013d}'
        first_col.append(str(tm[0]) + '_' + str(tm[1] + '_' + temp))

    first_col = np.array(first_col).reshape(-1, 1)
    first_col.tolist()
    a = pd.DataFrame(y_pred, columns=['floor', 'x', 'y'])
    a.insert(loc=0, column='site_path_timestamp', value=first_col)
    a.head()
    a.to_csv(base_path + 'output/submission_' + time.strftime("%Y%m%d-%H%M%S") + '.csv', index=False)


if __name__ == '__main__':
    base_path = '/media/kuro/Partition/'

    # =============================================================train==============================================

    # create data, save to csv
    dataset_path = base_path + 'train_kouki/'
    path_filenames = load_path_data(dataset_path, '.csv')
    feature_data = obtain_feature_data(path_filenames)

    feature_data = feature_data[:, 1:306]

    wifi_bssdi = collect_bssdi()
    scaler_bssdi, bssdi = encode_feature(wifi_bssdi)

    feature_data = select_best_feature(feature_data)
    np.savetxt(base_path + "train_npy/feature_data_hex.csv", feature_data, delimiter=",", fmt='%s')

    # load data from csv
    df = read_csv(base_path + "train_npy/feature_data_hex.csv", header=None, )
    feature_data = df.to_numpy()

    x, y = split_x_y(feature_data)
    y[:, 0] = y[:, 0] + 2

    scaler_building, x[:, 0] = encode_feature(x[:, 0])
    scaler_path, x[:, 1] = encode_feature(x[:, 1])
    scaler_bssdi, x[:, 2] = encode_feature(x[:, 2], scaler_bssdi)
    scaler_bssdi, x[:, 2] = encode_feature(x[:, 2])

    max_values, x = normalize_x(x)
    x = x.astype(float)

    feature_number = x.shape[1]
    floor_number = int(np.max(y[:, 0]) + 1)

    # model = train_floor_only()
    # model = train_loc_only()
    model = train_combined()

    model.save('model/' + time.strftime("%Y%m%d-%H%M%S") + '.h5')
    model.save('model/combined.h5')

    # # #=============================================================test==============================================

    # # create data, save to csv
    dataset_path = base_path + 'test_kouki/'
    path_filenames = load_path_data(dataset_path, '.csv')
    feature_data_test_ori = obtain_feature_data(path_filenames, train=False)

    feature_data_test = feature_data_test_ori[:, 2:307]
    feature_data_test = select_best_feature(feature_data_test)
    np.savetxt(base_path + "train_npy/feature_data_hex_test.csv", feature_data_test, delimiter=",", fmt='%s')

    # # load data from csv
    df_test = read_csv(base_path + "train_npy/feature_data_hex_test.csv", header=None)
    feature_data_test = df_test.to_numpy()

    x_test, y_test = split_x_y(feature_data_test)
    y_test[:, 0] = y_test[:, 0] + 2

    y_pred = np.load(base_path + 'output/pred.npy')
    export_submission(y_pred)
