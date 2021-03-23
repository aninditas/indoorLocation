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
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import time

def load_path_data(dataset_path,extension):
    path_filenames = list(Path(dataset_path).glob("**/*"+extension))
    return path_filenames

def obtain_feature_data(path_filenames, train=True):
    # 8 dimension: building, path, time, uuid, major, minor, rssi, floor   
    dimension = (11 if train==True else 10)
    feature_data = np.empty((0,dimension)) 
    waypoint_data = np.empty((0,6))
    for file_txt in path_filenames:
        path = Path(file_txt).name[:-4]
        floor = Path(file_txt).parent.name
        with open(file_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line_data in lines:
            line_data = line_data.strip()

            line_data = line_data.split('\t')
            if line_data[1][:6] == 'SiteID':
                building = line_data[1][7:]
            if line_data[1] == 'TYPE_WAYPOINT':
                time = line_data[0]
                x_loc = line_data[2]
                y_loc = line_data[3]
                waypoint_data = np.vstack((waypoint_data,(building, path, time, x_loc, y_loc, floor)))
            if line_data[1] == 'TYPE_BEACON':
                time = line_data[0]
                uuid = line_data[2]
                major = line_data[3]
                minor = line_data[4]
                val1 = line_data[5]
                val2 = line_data[6]
                val3 = line_data[7]
                val4 = line_data[8]
                if train:
                    feature_data = np.vstack((feature_data,(building, path, time, uuid, major, minor, val1, val2, val3, val4, floor)))
                else:
                    feature_data = np.vstack((feature_data,(building, path, time, uuid, major, minor, val1, val2, val3, val4)))
        np.savetxt(base_path+"dataset/train_npy/"+building+"_"+floor+"_"+path+"_"+"f.csv", feature_data, delimiter=",", fmt='%s')
        np.savetxt(base_path+"dataset/train_npy/"+building+"_"+floor+"_"+path+"_"+"w.csv", waypoint_data, delimiter=",", fmt='%s')
        print("Done... "+building+"_"+floor+"_"+path)
        feature_data = np.empty((0,dimension)) 
        waypoint_data = np.empty((0,6))
    # if train:
    #     # add 2 dimension: x_loc and y_loc from the nearest time
    #     feature_data = pair_waypoint_data(feature_data, waypoint_data)
    # return feature_data

def join_data(path_filenames, train=True):
   # 8 dimension: building, path, time, uuid, major, minor, rssi, floor   
    dimension = (11 if train==True else 10)
    feature_data = np.empty((0,dimension)) 
    waypoint_data = np.empty((0,6))
    for file_npy in path_filenames:
        try:
            df = read_csv(file_npy)
            temp = df.to_numpy()
            if Path(file_npy).name[-5]=='f':
                feature_data = np.vstack((feature_data,temp))
            else:
                waypoint_data = np.vstack((waypoint_data,temp))
            print("Saved to file "+Path(file_npy).name)
        except:
            continue
       
    return feature_data, waypoint_data
        
def pair_waypoint_data(feature_data, waypoint_data):
    new_feature_data = np.empty((0,13)) 
    for idx, feature_row in enumerate(feature_data):
        path = waypoint_data[:,1]==feature_row[1]
        floor = waypoint_data[:,-1]==feature_row[-1]
        building = waypoint_data[:,0]==feature_row[0]
        masked_waypoint = waypoint_data[(building)&(floor)&(path)]
        distance_waypoint = abs(masked_waypoint[:,2]-feature_row[2])
        try:
            shortest_distance = np.argmin(distance_waypoint)
            feature_row = np.append(feature_row,masked_waypoint[shortest_distance,3:5])
            new_feature_data = np.vstack((new_feature_data,feature_row))
        except:
            continue
        print(idx)
    return new_feature_data

def encode_floor(y):
    max_floor = 11
    new_y = []
    for yy in y:
        if 'F' in yy:
            try:
                new_y.append(int(yy.replace('F',''))-1)
            except:
                new_y.append(0)
        elif 'L' in yy:
            try:
                new_y.append(int(yy.replace('L',''))-1)
            except:
                new_y.append(0)        
        elif 'B' in yy:
            try:
                new_y.append(int(yy.replace('B',''))*-1)
            except:
                new_y.append(0)
        elif 'P' in yy:
            try:
                new_y.append(int(yy.replace('P',''))*-1)
            except:
                new_y.append(0)
        else: new_y.append(0)
    return new_y

# def encode_floor(y):
#     scaler = OrdinalEncoder()
#     y_floor = y[:,0].reshape(-1,1)
#     scaler.fit(y_floor)
#     return scaler, scaler.transform(y_floor)

def split_train_val_test(x, y, val_size = 0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test 

def split_x_y(paired_data):
    x = paired_data[:,:-3].astype(float)
    y_loc = paired_data[:,-2:].astype(float)
    y_floor = encode_floor(paired_data[:,-3])
    y_floor = np.reshape(y_floor,((-1,1)))
    y = np.hstack((y_floor,y_loc))
    print(x.shape, y.shape)
    return x,y

def build_model(floor_model):
    model = Sequential()
    model.add(Dense(128, input_dim=10, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(floor_number, activation='softmax')) 
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
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

def preprocessing_1():
    # obtain TYPE_BEACON data from text, save to csv file each path
    data_train_path = base_path+'dataset/train/'
    path_train_filenames = load_path_data(data_train_path,'.txt')
    obtain_feature_data(path_train_filenames)
    
    # join all csv files into one file
    data_npy_path = base_path+'dataset/train_npy/'
    path_npy_filenames = load_path_data(data_npy_path,'.csv')
    feature_data, waypoint_data = join_data(path_npy_filenames, train=True)
    
    # save the joined file
    np.savetxt(base_path+"dataset/train_npy/feature_data_npy.csv", feature_data, delimiter=",", fmt='%s')
    np.savetxt(base_path+"dataset/train_npy/waypoint_data_npy.csv", waypoint_data, delimiter=",", fmt='%s')
    
    feature_data, waypoint_data = convert_hex_to_int(feature_data, waypoint_data)
    
    # pair the feature_data with waypoint_data
    paired_data = pair_waypoint_data(feature_data, waypoint_data)
    np.savetxt(base_path+"dataset/train_npy/paired_data_npy.csv", paired_data, delimiter=",", fmt='%s')

def preprocessing_2():
    paired_data = read_csv(base_path+"dataset_npy/train_paired_data_npy.csv")
    paired_data = paired_data.to_numpy()
    max_values, paired_data = encode_data(paired_data)
    x,y = split_x_y(paired_data)
    return paired_data, max_values, x, y
    
def encode_data(paired_data):
    np.seterr(divide='ignore', invalid='ignore')
    max_values = [] 
    for idx in range(paired_data.shape[1]-3):
        paired_data[:,idx] = paired_data[:,idx].astype(float)
        max_values.append(paired_data[:,idx].max())
        paired_data[:,idx] = np.true_divide(paired_data[:,idx],max_values[idx])
    return max_values, paired_data

def convert_hex_to_int(feature_data, waypoint_data):
    idx = [0,1,3,4,5,9]
    for idxx in idx:
        feature_data[:,idxx]=[int(value, 16) for value in feature_data[:,idxx]]
    idx = [0,1]
    for idxx in idx:
        waypoint_data[:,idxx]=[int(value, 16) for value in waypoint_data[:,idxx]]
    return feature_data, waypoint_data

if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/'
    
    # # preprocessing_1 is for generating npy files
    # # once the npy files is generated, it does not have to be executed again
    # preprocessing_1()
    
    # preprocessing_2 is for loading the npy file
    # followed by encoding the data, and splitting it into x, y, train, val, and test
    paired_data, max_values, x, y = preprocessing_2()
    
    floor_number = np.max(y[:,0])+1
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)
    
    model = build_model(floor_number)
    history = model.fit(x_train,y_train[:,0], validation_data=(x_val,y_val[:,0]), epochs=10, batch_size=256)
    # model.fit(x_train,(y_train[:,0],y_train[:,1:]), validation_data=(x_val,(y_val[:,0],y_val[:,1:])), epochs=10, batch_size=128)
    
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
    
    print('evaluate')
    model.evaluate(x_test, (y_test[:,0],y_test[:,1:]))
    model.save('model/'+time.strftime("%Y%m%d-%H%M%S")+'.h5')
    
    # data_test_path = base_path+'dataset/test/'
    # path_test_filenames = load_path_data(data_test_path)
    # x_pred = obtain_feature_data(path_test_filenames,train=False).astype(float)

    # y_pred_floor, y_pred_loc = model.predict(x_pred)
    
    