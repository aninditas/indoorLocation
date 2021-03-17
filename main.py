# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:38 2021

@author: anindita
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import tensorflow as tf
import time

def load_path_data(dataset_path):
    path_filenames = list(Path(dataset_path).glob("**/*.txt"))
    return path_filenames

def obtain_feature_data(path_filenames, train=True):
    # 8 dimension: building, path, time, uuid, major, minor, rssi, floor
    
    dimension = (11 if train==True else 10)
    feature_data = np.empty((0,dimension)) 
    waypoint_data = np.empty((0,6))
    for file_txt in path_filenames:
        path = int(Path(file_txt).name[:-4],16)
        floor = Path(file_txt).parent.name
        # building = int(Path(file_txt).parent.parent.name,16)
        with open(file_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line_data in lines:
            line_data = line_data.strip()
            # if not line_data or line_data[0] == '#':
            #     continue
            line_data = line_data.split('\t')
            if line_data[1][:6] == 'SiteID':
                building = int(line_data[1][7:],16)
            if line_data[1] == 'TYPE_WAYPOINT':
                time = int(line_data[0])
                x_loc = float(line_data[2])
                y_loc = float(line_data[3])
                waypoint_data = np.vstack((waypoint_data,(building, path, time, x_loc, y_loc, floor)))
            if line_data[1] == 'TYPE_BEACON':
                time = int(line_data[0])
                uuid = int(line_data[2],16)
                major = int(line_data[3],16)
                minor = int(line_data[4],16)
                val1 = int(line_data[5])
                val2 = int(line_data[6])
                val3 = float(line_data[7])
                val4 = int(line_data[8],16)
                if train:
                    feature_data = np.vstack((feature_data,(building, path, time, uuid, major, minor, val1, val2, val3, val4, floor)))
                else:
                    feature_data = np.vstack((feature_data,(building, path, time, uuid, major, minor, val1, val2, val3, val4)))
    if train:
        # add 2 dimension: x_loc and y_loc from the nearest time
        feature_data = pair_waypoint_data(feature_data, waypoint_data)
    return feature_data

def pair_waypoint_data(feature_data, waypoint_data):
    new_feature_data = np.empty((0,13)) 
    for feature_row in feature_data:
        path = waypoint_data[:,1]==feature_row[1]
        floor = waypoint_data[:,-1]==feature_row[-1]
        building = waypoint_data[:,0]==feature_row[0]
        masked_waypoint = waypoint_data[(building)&(floor)&(path)]
        distance_waypoint = abs(masked_waypoint[:,2]-feature_row[2])
        shortest_distance = np.argmin(distance_waypoint)
        feature_row=np.append(feature_row,masked_waypoint[shortest_distance,3:5])
        new_feature_data = np.vstack((new_feature_data,feature_row))
    return new_feature_data

# def encode_data():
#     scalerX = OrdinalEncoder()
#     scalerX.fit(x)
#     scaled_x = scalerX.transform(x)
#     scaled_x[:,2] = x[:,2]
    
#     scalerY = OrdinalEncoder()
#     y_floor = y[:,0].reshape(-1,1)
#     scalerY.fit(y_floor)
#     scaled_y = scalerY.transform(y_floor)
#     scaled_y = np.hstack((scaled_y,y[:,-2:].astype(float)))
#     scaled_y[:,2] = y[:,2]
    
#     return scalerX, scalerY, scaled_x, scaled_y

# def encode_transform(scaler, data):
#     scaled_data = scaler.transform(data)
#     scaled_data[:,2] = data[:,2]
#     return scaled_data

def encode_floor(y):
    scaler = OrdinalEncoder()
    y_floor = y[:,0].reshape(-1,1)
    scaler.fit(y_floor)
    return scaler, scaler.transform(y_floor)

def split_train_val_test(x, y, val_size = 0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test 

def split_x_y(feature_data):
    x = feature_data[:,:-3].astype(float)
    y_loc = feature_data[:,-2:].astype(float)
    scaler, y_floor = encode_floor(feature_data[:,-3:])
    # y_floor = y_floor.squeeze()
    y = np.hstack((y_floor,y_loc))
    print(x.shape, y.shape)
    return scaler,x,y

def build_model():
    model = Sequential()
    model.add(Dense(12, input_dim=10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='linear'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model

# def test_model():
    
    

if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/'
    
    data_train_path = base_path+'dataset/sample_train/'
    path_train_filenames = load_path_data(data_train_path)
    feature_train = obtain_feature_data(path_train_filenames)
    scaler,x,y = split_x_y(feature_train)
    
    # scalerX, scalerY, x, y = encode_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)
    
    model = build_model()
    model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs=10, batch_size=128)
    print('evaluate')
    model.evaluate(x_test, y_test)
    # model.save('model/'+time.strftime("%Y%m%d-%H%M%S")+'.h5')
    
    data_test_path = base_path+'dataset/sample_test/'
    path_test_filenames = load_path_data(data_test_path)
    x_pred = obtain_feature_data(path_test_filenames,train=False).astype(float)
    # feature_test_scaled = scalerX.transform(feature_test)
    # feature_test_scaled[:,2] = feature_test[:,2]
    y_pred = model.predict(x_pred)
    
    