# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:38 2021

@author: anindita
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
# from numpy import genfromtxt
from pandas import read_csv
import pandas as pd
import numpy as np
import tensorflow as tf
import time

def load_path_data(dataset_path,extension):
    path_filenames = list(Path(dataset_path).glob("**/*"+extension))
    return path_filenames

def obtain_feature_data(path_filenames, train=True):
    col = 336 if train==True else 337
    feature_data = np.empty((0,col)) 
    for file_csv in path_filenames:
        df = read_csv(file_csv)
        temp = df.to_numpy()
        feature_data = np.vstack((feature_data, temp))
    return feature_data    

def split_train_val_test(x, y, val_size = 0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test 

def build_model_floor():
    # floor, 100 epoch 
    model = Sequential() 
    model.add(Dense(64, input_shape=(feature_number,), activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(floor_number, activation='softmax')) 
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def build_model_loc():
    # loc, 100 epoch
    model = Sequential() 
    model.add(Dense(64, input_shape=(feature_number,), activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(2, activation='linear')) 
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model

def build_model():
    inputs = Input(shape=(feature_number,), name='inputs')
    dense1 = Dense(64, activation='sigmoid', name='dense1')(inputs)
    dense2 = Dense(64, activation='sigmoid', name='dense2')(dense1)
    dense3 = Dense(64, activation='sigmoid', name='dense3')(dense2)
    
    dense4 = Dense(64, activation='sigmoid', name='dense4')(inputs)
    dense5 = Dense(64, activation='sigmoid', name='dense5')(dense4)
    dense6 = Dense(64, activation='sigmoid', name='dense6')(dense5)
    
    output_floor = Dense(floor_number, activation='softmax', name='denseFloor')(dense6)
    output_loc = Dense(2, activation='linear', name='denseLoc')(dense3)
    
    model = Model(inputs=inputs, outputs=[output_floor,output_loc])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=['sparse_categorical_crossentropy','mean_squared_error'], optimizer=opt, metrics=['accuracy'])
    model.summary()
    plot_model(model)
    return model

def convert_hex_to_int(data):
    idx = [0,2,5]
    for idxx in idx:
        try:
            data[:,idxx]=[int(value, 16) for value in data[:,idxx]]
        except:
            continue
    return data

# def select_best_feature(train=True):
#     distance_location = np.arange(7,306,3)
#     distance = feature_data[:,distance_location]
#     min_distance = np.argmin(distance, axis=1)
#     feature_min_time = np.empty((0,3))
#     for idx,row in enumerate(feature_data):
#         feature_min_time = np.vstack((feature_min_time,row[distance_location[min_distance[idx]]-2:distance_location[min_distance[idx]]+1]))
#     feature_min_time = np.hstack((feature_data[:,0:5],feature_min_time))
#     feature_clear = feature_min_time[feature_min_time[:,6]<3000]
#     # feature_int = convert_hex_to_int(feature_clear)
#     return feature_clear

def select_best_feature(feature_data, train=True):
    signal_location = np.arange(6,306,3)
    signal = feature_data[:,signal_location]
    max_signal = np.argmax(signal, axis=1)
    feature_max_signal = np.empty((0,3))
    for idx,row in enumerate(feature_data):
        feature_max_signal = np.vstack((feature_max_signal,row[signal_location[max_signal[idx]]-1:signal_location[max_signal[idx]]+2]))
    feature_max_signal = np.hstack((feature_data[:,0:5],feature_max_signal))
    # feature_clear = feature_max_signal[feature_max_signal[:,7]>-999]
    return feature_max_signal

def collect_bssdi():
    bssdi_location = np.arange(5,305,3)
    bssdi = []
    for b in bssdi_location:
        bssdi.append(feature_data[:,b])
    bssdi = np.unique(bssdi)
    return bssdi

def split_x_y(feature_data):
    x = feature_data[:,[0,2,5,6,7]]
    y = feature_data[:,[1,3,4]]
    return x,y

def encode_feature(feature, scaler=None):
    if scaler==None:
        scaler = OrdinalEncoder()
    feature = feature.reshape(-1,1)
    scaler.fit(feature)
    feature = scaler.transform(feature)
    feature = feature.squeeze()
    return scaler, feature.astype(float)

# def encode_x(x, scalers=None, train=True):
#     idx = [0,1,2]
#     if train==True:
#         scalers = []
#         for idxx in idx:
#             scalers.append(OrdinalEncoder())
#             temp = x[:,idxx].reshape(-1,1)
#             temp = temp.astype(str)
#             scalers[idxx].fit(temp)
#             temp = scalers[idxx].transform(temp)
#             x[:,idxx] = temp.squeeze()
#     else:
#         for idxx in idx:
#             temp = x[:,idxx].reshape(-1,1)
#             temp = temp.astype(str)
#             temp = scalers[idxx].transform(temp)
#             x[:,idxx] = temp.squeeze()
#     return scalers, x.astype(float)

# def decode_x(scalers):
#     idx = [0,1,2]
#     for idxx in idx:
#         temp = x[:,idxx].reshape(-1,1)
#         temp = scalers[idxx].inverse_transform(temp)
#         x[:,idxx] = temp.squeeze()
#     return x
    
# def encode_x():
#     max_values=[]
#     for col in range(x.shape[1]):
#         max_values.append(np.max(np.absolute(x[:,col])))
#         x[:,col] = x[:,col]/max_values[col]
#     return max_values, np.absolute(x)               

def normalize_x(x, max_values=None, train=True):
    if train==True:
        max_values=[]
        for col in range(x.shape[1]):
            max_values.append(np.max(np.absolute(x[:,col])))
            x[:,col] = x[:,col]/max_values[col]
    else:
        for col in range(x.shape[1]):
            x[:,col] = x[:,col]/max_values[col]
    return max_values, np.absolute(x)    

def train_floor_only():
    Y = y[:,0]
    Y = y.astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.1)
    
    model = build_model_floor()
    model.summary()
    history = model.fit(x_train,y_train, validation_split=0.1, epochs=100, batch_size=32)
    
    print('evaluate')
    model.evaluate(x_test, y_test)
    
    y_pred = model.predict(x_train)
    y_pred_arg = np.argmax(y_pred, axis=1)
    conf_mat = confusion_matrix(y_train, y_pred_arg)
    print('confusion matrix of training')
    print(conf_mat)
        
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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
    
    return model
    
def train_loc_only():
    Y = y[:,1:]
    Y = y.astype(int)
    
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.1)
    
    model = build_model_loc()
    model.summary()
    history = model.fit(x_train,y_train, validation_split=0.1, epochs=100, batch_size=32)
    
    print('evaluate')
    model.evaluate(x_test, y_test)
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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
    
    return model

def train_combined():
    Y = y.astype(float)
    
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.1)
    
    model = build_model()
    model.summary()
    history = model.fit(x_train, [y_train[:,0],y_train[:,1:]], validation_split=0.1, epochs=100, batch_size=32)
    
    print('evaluate')
    model.evaluate(x_test,  [y_test[:,0],y_test[:,1:]])
    
    # summarize history for accuracy
    plt.plot(history.history['denseFloor_accuracy'])
    plt.plot(history.history['denseLoc_accuracy'])
    plt.plot(history.history['val_denseFloor_accuracy'])
    plt.plot(history.history['val_denseLoc_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train floor', 'train loc', 'val floor', 'val loc'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['denseFloor_loss'])
    plt.plot(history.history['denseLoc_loss'])
    plt.plot(history.history['val_denseFloor_loss'])
    plt.plot(history.history['val_denseLoc_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train floor', 'train loc', 'val floor', 'val loc'], loc='upper left')
    plt.show()
    
    return model


    
    

if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/'
    
    #=============================================================train==============================================
    
    # create data, save to csv
    dataset_path = base_path+'dataset/train_kouki/'
    path_filenames = load_path_data(dataset_path,'.csv')
    feature_data = obtain_feature_data(path_filenames)
    
    feature_data = feature_data[:,1:306]
    
    wifi_bssdi = collect_bssdi()
    scaler_bssdi, bssdi = encode_feature(wifi_bssdi)
    
    feature_data = select_best_feature(feature_data)
    np.savetxt(base_path+"dataset/train_npy/train_path_npy_wifi/feature_data_hex.csv", feature_data, delimiter=",", fmt='%s')
    
    # load data from csv
    df = read_csv(base_path+"dataset/train_npy/train_path_npy_wifi/feature_data_hex.csv",header=None,)
    feature_data = df.to_numpy()
    
    x,y = split_x_y(feature_data)
    y[:,0] = y[:,0]+2
    
   
    scaler_building, x[:,0] = encode_feature(x[:,0])
    scaler_path, x[:,1] = encode_feature(x[:,1])
    scaler_bssdi, x[:,2] = encode_feature(x[:,2], scaler_bssdi)
    
    max_values, x = normalize_x(x)
    x = x.astype(float)
    
    feature_number = x.shape[1]
    floor_number = int(np.max(y[:,0])+1)
    
    # # model = train_floor_only()
    # # model = train_loc_only()
    model = train_combined()
    
    # model.save('model/'+time.strftime("%Y%m%d-%H%M%S")+'.h5')
    # model.save('model/combined.h5')
    
    
    #=============================================================test==============================================
    
    # create data, save to csv
    dataset_path = base_path+'dataset/test_kouki/'
    path_filenames = load_path_data(dataset_path,'.csv')
    feature_data_test_ori = obtain_feature_data(path_filenames, train=False)
    
    feature_data_test = feature_data_test_ori[:,2:307]
    feature_data_test = select_best_feature(feature_data_test)
    np.savetxt(base_path+"dataset/train_npy/train_path_npy_wifi/feature_data_hex_test.csv", feature_data_test, delimiter=",", fmt='%s')
    
    # load data from csv
    df_test = read_csv(base_path+"dataset/train_npy/train_path_npy_wifi/feature_data_hex_test.csv", header=None)
    feature_data_test = df_test.to_numpy()
    
    x_test,y_test = split_x_y(feature_data_test)
    y_test[:,0] = y_test[:,0]+2
    
    scaler_building, x_test[:,0] = encode_feature(x_test[:,0],scaler_building)
    scaler_path, x_test[:,1] = encode_feature(x_test[:,1],scaler_path)
    scaler_bssdi, x_test[:,2] = encode_feature(x_test[:,2],scaler_bssdi)
    
    max_values, x_test = normalize_x(x_test, max_values, train=False)
    
    feature_number = x.shape[1]
    floor_number = int(np.max(y[:,0])+1)
    
    # model = load_model('model/20210330-024104.h5')
    y_pred = model.predict(x_test)
    y_pred_floor = np.argmax(y_pred[0], axis=1)-2
    y_pred_loc = y_pred[1]
    
    first_col=[]
    for tm in feature_data_test_ori[:,[2,4,1]]:
        temp = f'{tm[2]:013d}'
        first_col.append(str(tm[0])+'_'+str(tm[1]+'_'+temp))
    
    first_col = np.array(first_col).reshape(-1,1)
    y_pred_floor = y_pred_floor.reshape(-1,1)
    y_export = np.hstack((first_col,y_pred_floor))
    y_export = np.hstack((y_export,y_pred_loc))
    y_export = pd.DataFrame({'site_path_timestamp': y_export[:, 0], 'floor': y_export[:, 1], 'x': y_export[:, 2], 'y': y_export[:, 3]})
    y_export.to_csv(base_path+'output/submission_'+time.strftime("%Y%m%d-%H%M%S")+'.csv', index=False)
    # np.savetxt(base_path+'output/submission_'+time.strftime("%Y%m%d-%H%M%S")+'.csv', y_export, delimiter=",", fmt='%s')
    
    # issues: cannot decode wifi bssdi --> possibly because they aren't selected
    # solution: encode all bssdi even if not selected
    
    
    
    
    
    
    
    
    