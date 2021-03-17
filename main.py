# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:38 2021

@author: anindita
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def load_path_data():
    path_filenames = list(Path(dataset_path).glob("**/*.txt"))
    return path_filenames

def obtain_feature_data():
    # 8 dimension: building, path, time, uuid, major, minor, rssi, floor
    feature_data = np.empty((0,8)) 
    waypoint_data = np.empty((0,6))
    for file_txt in path_filenames[0:2]:
        path = Path(file_txt).name[:-4]
        floor = Path(file_txt).parent.name
        building = Path(file_txt).parent.parent.name
        with open(file_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line_data in lines:
            line_data = line_data.strip()
            if not line_data or line_data[0] == '#':
                continue
            line_data = line_data.split('\t')
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
                rssi = line_data[6]
                feature_data = np.vstack((feature_data,(building, path, time, uuid, major, minor, rssi, floor)))
    # add 2 dimension: x_loc and y_loc from the nearest time
    feature_data = pair_waypoint_data(feature_data, waypoint_data)
    return feature_data

def pair_waypoint_data(feature_data, waypoint_data):
    new_feature_data = np.empty((0,10)) 
    for feature_row in feature_data:
        path = waypoint_data[:,1]==feature_row[1]
        floor = waypoint_data[:,5]==feature_row[7]
        building = waypoint_data[:,0]==feature_row[0]
        masked_waypoint = waypoint_data[(building)&(floor)&(path)]
        distance_waypoint = abs(masked_waypoint[:,2].astype(float)-feature_row[2].astype(float))
        shortest_distance = np.argmin(distance_waypoint)
        feature_row=np.append(feature_row,masked_waypoint[shortest_distance,3:5])
        new_feature_data = np.vstack((new_feature_data,feature_row))
    return new_feature_data

def split_train_val_test(x, y, val_size = 0.1, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, y_train, x_val, y_val, x_test, y_test 

def split_x_y():
    x = feature_data[:,:7]
    y = feature_data[:,7:]
    print(x.shape, y.shape)
    return x,y

if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/dataset/'
    dataset_path = base_path+'train/'
    path_filenames=load_path_data()
    feature_data = obtain_feature_data()
    x,y = split_x_y()
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)
    