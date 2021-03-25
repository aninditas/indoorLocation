# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:38 2021

@author: anindita
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from numpy import genfromtxt
from pandas import read_csv
import numpy as np
import tensorflow as tf
import time

def load_path_data(dataset_path,extension):
    path_filenames = list(Path(dataset_path).glob("**/*"+extension))
    return path_filenames

def obtain_feature_data(path_filenames):
    feature_data = np.empty((0,336)) 
    for file_csv in path_filenames:
        df = read_csv(file_csv)
        temp = df.to_numpy()
        feature_data = np.vstack((feature_data, temp))
    return feature_data    

def split_train_val_test(x, y, val_size = 0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test 

def build_model():
    model = Sequential() 
    model.add(Dense(8, input_dim=feature_number, activation='relu'))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear')) 
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model


# def build_model():
#     inputs = Input(shape=(10,))
#     dense1 = Dense(12, activation='relu')(inputs)
#     dense2 = Dense(8, activation='relu')(dense1)
#     output_floor = Dense(floor_number, activation='softmax')(dense2)
#     output_loc = Dense(2, activation='linear')(dense2)
#     model = Model(inputs=inputs, outputs=(output_floor,output_loc))
#     opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
#     model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
#     model.summary()
#     # plot_model(model)
#     return model

def convert_hex_to_int(data):
    idx = [0,2,5]
    for idxx in idx:
        try:
            data[:,idxx]=[int(value, 16) for value in data[:,idxx]]
        except:
            continue
    return data

def select_best_feature():
    distance_location = np.arange(7,306,3)
    distance = feature_data[:,distance_location]
    min_distance = np.argmin(distance, axis=1)
    feature_min_time = np.empty((0,3))
    for idx,row in enumerate(feature_data):
        feature_min_time = np.vstack((feature_min_time,row[distance_location[min_distance[idx]]-2:distance_location[min_distance[idx]]+1]))
    feature_min_time = np.hstack((feature_data[:,0:5],feature_min_time))
    feature_clear = feature_min_time[feature_min_time[:,7]<3000]
    feature_int = convert_hex_to_int(feature_clear)
    return feature_int

def split_x_y():
    x = feature_data[:,[0,2,5,6,7]].astype(float)
    y = feature_data[:,[1,3,4]]
    return x,y.astype(float)

# def encode_x():
#     scalers = []
#     idx = [0,1,2]
#     for idxx in idx:
#         scalers.append(OrdinalEncoder())
#         temp = x[:,idxx].reshape(-1,1)
#         scalers[idxx].fit(temp)
#         temp = scalers[idxx].transform(temp)
#         x[:,idxx] = temp.squeeze()
#     return scalers, x.astype(float)
    

def encode_x():
    max_values=[]
    for col in range(x.shape[1]):
        max_values.append(np.max(np.absolute(x[:,col])))
        x[:,col] = x[:,col]/max_values[col]
    return max_values, np.absolute(x)                       

if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/'
    
    # # create data, save to csv
    # dataset_path = base_path+'dataset/train_kouki/'
    # path_filenames = load_path_data(dataset_path,'.csv')
    # feature_data = obtain_feature_data(path_filenames)
    
    # feature_data = feature_data[:,1:306]
    # feature_data = select_best_feature()
    # np.savetxt(base_path+"dataset/train_kouki/feature_data.csv", feature_data, delimiter=",", fmt='%s')
    
    # load data from csv
    df = read_csv(base_path+"dataset/train_kouki/feature_data.csv")
    feature_data = df.to_numpy()
    
    x,y = split_x_y()
    y[:,0] = y[:,0]+2
    scalers, x = encode_x()
    
    feature_number = x.shape[1]
    floor_number = int(np.max(y[:,0])+1)
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)
    
    model = build_model()
    model.fit(x_train,y_train[:,1:], validation_data=(x_val,y_val[:,1:]), epochs=10, batch_size=32)
    print('evaluate')
    model.evaluate(x_test, (y_test[:,0],y_test[:,1:]))
    # model.save('model/'+time.strftime("%Y%m%d-%H%M%S")+'.h5')
    
    # data_test_path = base_path+'dataset/test/'
    # path_test_filenames = load_path_data(data_test_path)
    # x_pred = obtain_feature_data(path_test_filenames,train=False).astype(float)

    # y_pred_floor, y_pred_loc = model.predict(x_pred)
    
    