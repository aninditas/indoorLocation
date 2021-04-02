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
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from pandas import read_csv
import pandas as pd
import numpy as np
import tensorflow as tf
import time

def obtain_feature_data(file_csv, train=True):
    df = read_csv(file_csv)
    feature_data = df.to_numpy()
    feature_data = feature_data[:,1:]
    return feature_data    

def build_model():
    inputs = Input(shape=(feature_number,), name='inputs')
    dense1 = Dense(64, activation='relu', name='dense1')(inputs)
    dense2 = Dense(64, activation='relu', name='dense2')(dense1)
    dense3 = Dense(64, activation='relu', name='dense3')(dense2)
    
    dense4 = Dense(64, activation='relu', name='dense4')(inputs)
    dense5 = Dense(64, activation='relu', name='dense5')(dense4)
    dense6 = Dense(64, activation='relu', name='dense6')(dense5)
    
    dense7 = Dense(64, activation='relu', name='dense7')(inputs)
    dense8 = Dense(64, activation='relu', name='dense8')(dense7)
    dense9 = Dense(64, activation='relu', name='dense9')(dense8)
    
    output_floor = Dense(floor_number, activation='softmax', name='denseFloor')(dense3)
    output_loc_X = Dense(1, activation='linear', name='denseLoc_X')(dense6)
    output_loc_Y = Dense(1, activation='linear', name='denseLoc_Y')(dense9)
    

    model = Model(inputs=inputs, outputs=[output_floor,output_loc_X,output_loc_Y])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=['sparse_categorical_crossentropy','mean_squared_error','mean_squared_error'], optimizer=opt, metrics=tf.keras.metrics.RootMeanSquaredError())
    # model.summary()
    plot_model(model)
    return model

def split_x_y(feature_data, train=True):
    x = feature_data[:,-1]
    x = x.reshape(-1,1)
    if train==True:
        x = np.hstack((x,feature_data[:,:-4]))
        y = feature_data[:,[-2,-4,-3]]
    else:
        x = np.hstack((x,feature_data[:,:-1]))
    return x,y

def encode_feature(feature, scaler=None):
    if scaler==None:
        scaler = OrdinalEncoder()
    feature = feature.reshape(-1,1)
    scaler.fit(feature)
    feature = scaler.transform(feature)
    feature = feature.squeeze()
    return scaler, feature.astype(float)            

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

def train(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    model = build_model()
    
    checkpoint_cb = ModelCheckpoint("/Model/cp_model.h5", save_best_only=True)
    history = model.fit(x_train, [y_train[:,0],y_train[:,1],y_train[:,2]], validation_split=0.1, epochs=100, batch_size=32, callbacks=[checkpoint_cb])
    
    print('evaluate')
    model.evaluate(x_test,  [y_test[:,0],y_test[:,1],y_test[:,2]])
    
    return model, history
 
if __name__ == '__main__':
    base_path = 'D:/Dropbox/PhD/python/IndoorLocation/IndoorLocation/'
    
    dataset_path = base_path+'dataset/train_devin/'
    path_filenames = list(Path(dataset_path).glob("**/*"+'.csv'))
    
    dataset_path_test = base_path+'dataset/test_devin/'
    path_filenames_test = list(Path(dataset_path_test).glob("**/*"+'.csv'))
    
    rmse=np.empty((0,3))
    models=[]
    building_path=[]
    y_pred_floor=[]
    y_pred_loc_x=[]
    y_pred_loc_y=[]
    
    
    for file_csv_train,file_csv_test in zip(path_filenames,path_filenames_test):
        
        # training process
        print(file_csv_train)
        feature_data_train = obtain_feature_data(file_csv_train)
        x,y = split_x_y(feature_data_train)
        y[:,0] = y[:,0]+2
        scaler_path, x[:,0] = encode_feature(x[:,0])
        max_values, x = normalize_x(x)
        
        x = x.astype(float)
        y = y.astype(float)
        
        feature_number = x.shape[1]
        floor_number = int(np.max(y[:,0])+1)
        model_, history = train(x,y)
        models = np.append(models,model_)
        rmse_floor = history.history['val_denseFloor_root_mean_squared_error'][-1]
        rmse_x = history.history['val_denseLoc_X_root_mean_squared_error'][-1]
        rmse_y = history.history['val_denseLoc_Y_root_mean_squared_error'][-1]
        rmse = np.vstack((rmse,(rmse_floor,rmse_x,rmse_y)))
        
        # testing process
        feature_data_test = obtain_feature_data(file_csv_test, train=False)
        x_test_ = feature_data_test[:,-1]
        x_test=[]
        for xt in x_test_:
            x_test.append(xt[25:49])
        x_test = np.array(x_test).reshape(-1,1)
        x_test = np.hstack((x_test,feature_data_test[:,:-1]))
        
        scaler_path, x_test[:,0] = encode_feature(x_test[:,0],scaler_path)
        x_test = x_test.astype(float)
        max_values, x_test = normalize_x(x_test, max_values, train=False)
        
        feature_number = x.shape[1]
        floor_number = int(np.max(y[:,0])+1)
        
        y_pred = model_.predict(x_test)
        temp = np.argmax(y_pred[0], axis=1)-2
        temp = temp.squeeze()
        building_path = np.append(building_path,x_test_)
        y_pred_floor = np.append(y_pred_floor,temp)
        y_pred_loc_x = np.append(y_pred_loc_x,y_pred[1])
        y_pred_loc_y = np.append(y_pred_loc_y,y_pred[2])
    
    building_path = np.array(building_path).reshape(-1,1)
    y_pred_floor = np.array(y_pred_floor).reshape(-1,1)
    y_pred_loc_x = np.array(y_pred_loc_x).reshape(-1,1)
    y_pred_loc_y = np.array(y_pred_loc_y).reshape(-1,1)
    
    y_export = np.hstack((building_path,y_pred_floor,y_pred_loc_x,y_pred_loc_y))
    
    y_export = pd.DataFrame({'site_path_timestamp': y_export[:, 0], 'floor': y_export[:, 1], 'x': y_export[:, 2], 'y': y_export[:, 3]})
    y_export.to_csv(base_path+'output/submission_'+time.strftime("%Y%m%d-%H%M%S")+'.csv', index=False)
    
    print(np.average(rmse,axis=0))
    
    
    
    
    