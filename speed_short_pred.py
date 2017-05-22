
# coding: utf-8

# In[2]:

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import concatenate, merge, Dense, LSTM, Input, Reshape, Convolution2D, Deconvolution2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.activations import relu, softmax, linear
from keras.layers.advanced_activations import PReLU, ELU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import math
# import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from pylab import *
from matplotlib import gridspec
cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.125, 1.0, 1.0),
                 (0.25, 1.0, 1.0),
                 (0.5625, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5625, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

# %matplotlib inline

#datapath = '/home/microway/Shuo/CarND/CarND-BehaviorCloning-Project/data-given/'


# In[ ]:

# def load_data(data,datafolder='Data_2016_DES_I235E/'):
#     X_train = np.zeros((len(data),15,288,10))
#     y_train = np.zeros((len(data),15,1440,3))
#     for i in range(len(data)):
#         with open(datafolder+'CSVs/'+data['X'][i], 'r') as f:
#             X = list(csv.reader(f, delimiter=","))
#             X = np.asarray(X)
#             channels = np.unique(X[:,0])
#             for channel in channels:
#                 index=X[:,0] == channel
#                 X_train[i,:,:,int(channel)] = X[index,1:]
#         with open(datafolder+'Traffic_CSVs/'+data['y'][i], 'r') as f:
#             y = list(csv.reader(f, delimiter=","))
#             y = np.asarray(y)
#             channels = np.unique(y[:,0])
#             for channel in channels:
#                 index=y[:,0] == channel
#                 y_train[i,:,:,int(channel)] = y[index,1:]
# #             index=y[:,0] == '0'
# #             y_train[i,:,:,0] = y[index,1:]
#     return X_train,y_train

# def convert_zero_2_mean(y_train):
#     y_train[y_train==0] = np.nan
#     avg = np.nanmean(y_train, axis=0)
#     arrays = [avg for _ in range(y_train.shape[0])]
#     AVG = np.stack(arrays, axis=0)
#     index = np.isnan(y_train)
#     y_train[index] = AVG[index]
#     y_train[np.isnan(y_train)] = np.nanmean(y_train)
#     return y_train




# In[ ]:

# data = pd.read_csv('/Users/Shuo/study/Project-predictive_study/data_2016_I235E.csv',delimiter=',')
# _,Traffic = load_data(data)
# Traffic = np.swapaxes(Traffic,1,2)
# Raw_Speed = Traffic[:,:,:,0]
# Speed = convert_zero_2_mean(Raw_Speed)
# Speed = np.reshape(Speed, (Speed.shape[0],Speed.shape[1], Speed.shape[2]))

# print(Traffic.shape)
# print(Raw_Speed.shape)
# print(Speed.shape)

# # print(Speed[16,:5,:])
# # plt.pcolor(Speed[16,:,:],cmap=my_cmap)
# print(np.count_nonzero(Speed==0))
# print(np.count_nonzero(np.isnan(Speed)))

# data['y'][0][:4]
# data['day'] = data['y'].map(lambda x: x[:8])
# data['date'] = pd.to_datetime(data['day'],format='%Y%m%d')
# data['dayofweek'] = data['date'].map(lambda x: x.dayofweek)
# data['dayofyear'] = data['date'].map(lambda x: x.dayofyear)
# del data['day']


# In[ ]:

# import pickle
# pickle.dump( (Traffic,Speed,data), open( "speed_short_term.p", "wb" ) )


# In[129]:

import pickle
(Traffic,Speed,data) = pickle.load( open( "speed_short_term.p", "rb" ) )
Count = Traffic[:,:,:,1]
Occup = Traffic[:,:,:,2]
print(Traffic.shape)
print(Speed.shape)
data.tail(5)


# In[150]:

def get_certain_dayofweek(Speed,dayofweek = 0):
    data_sub = data[:len(Speed)]
    Mon = Speed[data_sub.index[data_sub['dayofweek'] == dayofweek],:,:]
    mon=np.mean(Mon,axis=0)
    mon_std=np.std(Mon,axis=0)
    Mon_delta = Mon - mon
    mon_std[mon_std<0.1] = np.mean(mon_std)
    Mon_Z = Mon_delta/mon_std
    return Mon, Mon_delta, Mon_Z, mon, mon_std

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def MinMax_Normalization(samples):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0]*samples_shape[1]*samples_shape[2],1))
    scaler = MinMaxScaler().fit(samples)
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2]))
    return samples_normalized, scaler

def transfer_scale(samples,scaler):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0]*samples_shape[1]*samples_shape[2],1))
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2]))
    return samples_normalized
    

# Mon, Mon_delta, Mon_Z, mon_mean, mon_std = get_certain_dayofweek(Speed,dayofweek = 0)


# In[ ]:

# Mon, _, _, _, _ = get_certain_dayofweek(Speed,0)
# Tue, _, _, _, _ = get_certain_dayofweek(Speed,1)
# Wed, _, _, _, _ = get_certain_dayofweek(Speed,2)
# Thu, _, _, _, _ = get_certain_dayofweek(Speed,3)
# Fri, _, _, _, _ = get_certain_dayofweek(Speed,4)
# Sat, _, _, _, _ = get_certain_dayofweek(Speed,5)
# Sun, _, _, _, _ = get_certain_dayofweek(Speed,6)

# gs = gridspec.GridSpec(7, 5, wspace=0, hspace=0.15)

# fig = plt.figure(figsize=(15,22))

# ax = plt.subplot(gs[0,0])
# plt.plot(mon[:,0])
# ax = plt.subplot(gs[0,1])
# plt.plot(mon[:,7])
# ax = plt.subplot(gs[0,2])
# plt.plot(mon[:,14])
# ax = plt.subplot(gs[0,3:])
# plt.pcolor(np.swapaxes(mon,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Monday')
# colorbar()
# ax = plt.subplot(gs[1,0])
# plt.plot(tue[:,0])
# ax = plt.subplot(gs[1,1])
# plt.plot(tue[:,7])
# ax = plt.subplot(gs[1,2])
# plt.plot(tue[:,14])
# ax = plt.subplot(gs[1,3:])
# plt.pcolor(np.swapaxes(tue,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Tuesday')
# colorbar()
# ax = plt.subplot(gs[2,0])
# plt.plot(wed[:,0])
# ax = plt.subplot(gs[2,1])
# plt.plot(wed[:,7])
# ax = plt.subplot(gs[2,2])
# plt.plot(wed[:,14])
# ax = plt.subplot(gs[2,3:])
# plt.pcolor(np.swapaxes(wed,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Wednesday')
# colorbar()
# ax = plt.subplot(gs[3,0])
# plt.plot(thu[:,0])
# ax = plt.subplot(gs[3,1])
# plt.plot(thu[:,7])
# ax = plt.subplot(gs[3,2])
# plt.plot(thu[:,14])
# ax = plt.subplot(gs[3,3:])
# plt.pcolor(np.swapaxes(thu,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Thursday')
# colorbar()
# ax = plt.subplot(gs[4,0])
# plt.plot(fri[:,0])
# ax = plt.subplot(gs[4,1])
# plt.plot(fri[:,7])
# ax = plt.subplot(gs[4,2])
# plt.plot(fri[:,14])
# ax = plt.subplot(gs[4,3:])
# plt.pcolor(np.swapaxes(fri,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Friday')
# colorbar()
# ax = plt.subplot(gs[5,0])
# plt.plot(sat[:,0])
# ax = plt.subplot(gs[5,1])
# plt.plot(sat[:,7])
# ax = plt.subplot(gs[5,2])
# plt.plot(sat[:,14])
# ax = plt.subplot(gs[5,3:])
# plt.pcolor(np.swapaxes(sat,0,1),cmap=my_cmap, vmin=20, vmax=70)
# ax.set_title('Saturday')
# colorbar()
# ax = plt.subplot(gs[6,0])
# plt.plot(sun[:,0])
# ax = plt.subplot(gs[6,1])
# plt.plot(sun[:,7])
# ax = plt.subplot(gs[6,2])
# plt.plot(sun[:,14])
# ax = plt.subplot(gs[6,3:])
# plt.pcolor(np.swapaxes(sun,0,1),cmap=my_cmap, vmin=20, vmax=70)
# colorbar()
# ax.set_title('Sunday')
        
# fig.savefig('images/AvgSpeed_by_dayofweek.png')


# In[93]:

def shapeback(Y):
    YY = np.reshape(Y[len(Y)%288:,:],(len(Y)//288,288,Y.shape[1]))
    return np.swapaxes(YY,1,2)

def create_dataset(Speed, Speed_y, look_back=15, mode='uni'):
    
    dataX,dataY = [],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,0:1]
            dataset_y = Speed_y[j,:,0:1]

            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset_y[i + look_back, :])
                
            dataX.append(numpy.array(dataX_))
            dataY.append(numpy.array(dataY_))
            
    if mode == 'multi':
        
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,:]
            dataset_y = Speed_y[j,:,:]

            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset_y[i + look_back, :])
                
            dataX.append(numpy.array(dataX_))
            dataY.append(numpy.array(dataY_))    
                
                
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_historyAsFeature(Speed, Speed_y, look_back=15, mode='uni'):
    
    dataX,dataY = [],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)-look_back):
            dataX_,dataY_ = [],[]
            dataset = Speed[j + look_back,:,0:1]
            dataset_y = Speed_y[j + look_back,:,0:1]
            prevdata = Speed[j:(j + look_back),:,0:1]

            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i + look_back,:]
                dataX_.append(np.hstack([a,b]))
                dataY_.append(dataset_y[i + look_back, :])
                
            dataX.append(numpy.array(dataX_))
            dataY.append(numpy.array(dataY_))
            
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_historyAsSecondInput(Speed, Speed_y, look_back=15,look_back_days=6, mode='uni'):
    
    dataX1,dataX2,dataY = [],[],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)-look_back_days):
            dataX1_,dataX2_,dataY_ = [],[],[]
            dataset = Speed[j + look_back_days,:,0:1]
            dataset_y = Speed_y[j + look_back_days,:,0:1]
            prevdata = Speed[j:(j + look_back_days),:,0:1]

            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i+look_back,:]
                dataX1_.append(a)
                dataX2_.append(b)
                dataY_.append(dataset_y[i + look_back, :])
                
            dataX1.append(numpy.array(dataX1_))
            dataX2.append(numpy.array(dataX2_))
            dataY.append(numpy.array(dataY_))
            
    if mode == 'multi':
        
        for j in range(len(Speed)-look_back_days):
            dataX1_,dataX2_,dataY_ = [],[],[]
            dataset = Speed[j + look_back_days,:,:]
            dataset_y = Speed_y[j + look_back_days,:,:]
            prevdata = Speed[j:(j + look_back_days),:,:]

            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i+look_back,:]
                dataX1_.append(a)
                dataX2_.append(b)
                dataY_.append(dataset_y[i + look_back, :])
                
            dataX1.append(numpy.array(dataX1_))
            dataX2.append(numpy.array(dataX2_))
            dataY.append(numpy.array(dataY_))
            
    return numpy.array(dataX1), numpy.array(dataX2), numpy.array(dataY)


train_speed, _, _, _, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
test_speed = train_speed[-1:]
train_speed = train_speed[:-1]

print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)
look_back = 15
mode = 'uni'
train_speed_x,train_speed_y = create_dataset(train_speed,train_speed, look_back, mode)
test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)
print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[179]:

def history_plot(history_object,image1,image2,a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1))):
  
    trainScore = [];
    for i in range(len(train_speed_x)):
        trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x)):
        testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x[0,start,:,:]
    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    
    plt.plot(np.arange(look_ahead),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        trainPredict = test_speed_x[0,start+i,:,:]
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

def history_plot_historyAsFeature(history_object,image1,image2):
  
    trainScore = [];
    for i in range(len(train_speed_x)):
        trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x)):
        testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x[0,start,:,:]
    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.hstack([np.vstack([trainPredict[1:,:1],prediction]),test_speed_x[0,start+i+1,:,:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),0],label="test function")
    plt.legend()

    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        trainPredict = test_speed_x[0,start+i,:,:]
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),0],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

def history_plot_historyAsSecondInput(history_object,image1,image2,a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1))):
  
    trainScore = [];
    for i in range(len(train_speed_x1)):
        trainScore.append(model.evaluate([train_speed_x1[i,:,:,:],train_speed_x2[i,:,:,:]], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x1)):
        testScore.append(model.evaluate([test_speed_x1[i,:,:,:],test_speed_x2[i,:,:,:]], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x1[0,start,:,:]  
    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        input2 = test_speed_x2[0,start+i,:,:]
        prediction = model.predict([np.array([trainPredict]),np.array([input2])], batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    predictions = np.zeros((look_ahead,1))

    for i in range(look_ahead):
        trainPredict = test_speed_x1[0,start+i,:,:]
        input2 = test_speed_x2[0,start+i,:,:]
        prediction = model.predict([np.array([trainPredict]),np.array([input2])], batch_size=batch_size)
        predictions[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')


# In[26]:

epochs = 200


# ### Experiment1: input: univariate speed; output: univariate speed; lookback = 15; same day of week

# In[ ]:

# train_speed, _, _, _, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
# test_speed = train_speed[-1:]
# train_speed = train_speed[:-1]

# print('train_speed.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[ ]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset(train_speed,train_speed, look_back, mode)
# test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[ ]:

# batch_size = train_speed_x.shape[1]

# model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# # model.add(Dropout(0.3))
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
# # model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
# history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)


# In[ ]:

# history_plot(history,'images/history_exp1.png','images/test_exp1.png')


# ### Experiment1.1: input: univariate speed; output: univariate speed; lookback = 15; all year

# In[221]:

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:333,:,:]
# print('train_speed.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[222]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset(train_speed,train_speed, look_back, mode)
# test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# batch_size = train_speed_x.shape[1]

# model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# # model.add(Dropout(0.3))
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
# # model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
# history = model.fit(train_x, train_y,epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# # history_plot(history)


# In[ ]:

# history_plot(history,'images/history_exp11.png','images/test_exp11.png')


# ### Experiment2: input: univariate speed; output: univariate speed; lookback = 15; lookback weeks = 15 (as feature); consecutive previous days

# In[ ]:


# train_speed = Speed[:334,:,:]

# print('train_speed.shape = ',train_speed.shape)


# In[ ]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset_historyAsFeature(train_speed,train_speed, look_back, mode)
# # test_speed_x,test_speed_y = create_dataset_historyAsFeature(test_speed, look_back, mode)
# test_speed_x = train_speed_x[-1:,:,:,:]
# test_speed_y = train_speed_y[-1:,:,:]
# train_speed_x = train_speed_x[:-1,:,:,:]
# train_speed_y = train_speed_y[:-1,:,:]
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[ ]:

# batch_size = train_speed_x.shape[1]

# model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# # model.add(Dropout(0.3))
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
# # model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
# history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# # history_plot_historyAsFeature(history)


# In[ ]:

# history_plot_historyAsFeature(history,'images/history_exp2.png','images/test_exp2.png')


# ### Experiment2.1: input: univariate speed; output: univariate speed; lookback = 15; lookback weeks = 15 (as feature); previous same day of week

# In[ ]:

# # test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
# dayofweek = data['dayofweek'][333]
# train_speed = Speed[data.index[data['dayofweek'] == dayofweek],:,:]
# train_speed = train_speed[0:48,:,:]
# print('train_speed.shape = ',train_speed.shape)
# # print('test_speed.shape = ',test_speed.shape)


# In[ ]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset_historyAsFeature(train_speed,train_speed, look_back, mode)
# # test_speed_x,test_speed_y = create_dataset_historyAsFeature(test_speed, look_back, mode)
# test_speed_x = train_speed_x[-1:,:,:,:]
# test_speed_y = train_speed_y[-1:,:,:]
# train_speed_x = train_speed_x[:-1,:,:,:]
# train_speed_y = train_speed_y[:-1,:,:]
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[ ]:

# batch_size = train_speed_x.shape[1]

# model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# # model.add(Dropout(0.3))
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
# # model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
# history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# # history_plot_historyAsFeature(history)


# In[ ]:

# history_plot_historyAsFeature(history,'images/history_exp21.png','images/test_exp21.png')


# ### Experiment3: input: univariate speed; output: univariate speed; lookback = 15; lookback weeks = 6 (parallel structure); Monday only for training, Monday for testing
# 

# In[194]:

# train_speed, _, _, _, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
# # test_speed = train_speed[-1:]
# # train_speed = train_speed[:-1]

# print('train_speed.shape = ',train_speed.shape)
# # print('test_speed.shape = ',test_speed.shape)


# In[ ]:

# look_back = 15
# look_back_days = 6
# mode = 'uni'
# train_speed_x1,train_speed_x2,train_speed_y = create_dataset_historyAsSecondInput(train_speed,train_speed, look_back, look_back_days, mode)

# test_speed_x1 = train_speed_x1[-1:,:,:,:]
# test_speed_x2 = train_speed_x2[-1:,:,:,:]
# test_speed_y = train_speed_y[-1:,:,:]
# train_speed_x1 = train_speed_x1[:-1,:,:,:]
# train_speed_x2 = train_speed_x2[:-1,:,:,:]
# train_speed_y = train_speed_y[:-1,:,:]

# print('look_back = ',look_back)
# print('look_back_days = ',look_back_days)
# print('mode = ',mode)
# print('train_speed_x1.shape = ',train_speed_x1.shape)
# print('train_speed_x2.shape = ',train_speed_x2.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x1.shape = ',test_speed_x1.shape)
# print('test_speed_x2.shape = ',test_speed_x2.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[ ]:

# batch_size = train_speed_x.shape[1]


# todaySequence = Input(shape=(look_back, train_speed_x1.shape[3]),name='todaySequence')
# h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False, return_sequences=True)(todaySequence)
# h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False)(h1)

# historySequence = Input(shape=(look_back_days, train_speed_x2.shape[3]),name='historySequence')
# h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False, return_sequences=True)(historySequence)
# h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False)(h2)

# h3 = keras.layers.concatenate([h1, h2])
# predictedSpeed = Dense(1,name='predictedSpeed')(h3)

# model = Model(inputs=[todaySequence, historySequence], outputs=[predictedSpeed])

# model.compile(loss='mean_squared_error', optimizer='adam')

# # model.compile(optimizer='rmsprop',
# #               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
# #               loss_weights={'main_output': 1., 'aux_output': 0.2})

# train_x1 = np.reshape(train_speed_x1,(train_speed_x1.shape[0]*train_speed_x1.shape[1],train_speed_x1.shape[2],train_speed_x1.shape[3]))
# train_x2 = np.reshape(train_speed_x2,(train_speed_x2.shape[0]*train_speed_x2.shape[1],train_speed_x2.shape[2],train_speed_x2.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))

# history = model.fit({'todaySequence': train_x1, 'historySequence': train_x2},
#           {'predictedSpeed': train_y},
#           epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

# # history_plot_historyAsSecondInput(history)


# In[ ]:

# history_plot_historyAsSecondInput(history,'images/history_exp3.png','images/test_exp3.png')


# ### Experiment3.1: input: univariate speed; output: univariate speed; lookback = 15; lookback weeks = 6 (parallel structure); all weekdays for training, monday for testing
# 

# In[ ]:

# # test_speed = Speed[333:334,:,:]
# # train_speed = Speed[:334,:,:]
# # dayofweek = data['dayofweek'][333]
# train_speed0,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
# train_speed1,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
# train_speed2,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
# train_speed3,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
# train_speed4,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
# train_speed5,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
# train_speed6,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# # train_speed0 = train_speed0[0:48,:,:]
# print('train_speed0.shape = ',train_speed.shape)
# # print('test_speed.shape = ',test_speed.shape)


# In[ ]:

# look_back = 15
# look_back_days = 6
# mode = 'uni'
# train_speed_x1,train_speed_x2,train_speed_y = create_dataset_historyAsSecondInput(train_speed0,train_speed0, look_back, look_back_days, mode)

# test_speed_x1 = train_speed_x1[-1:,:,:,:]
# test_speed_x2 = train_speed_x2[-1:,:,:,:]
# test_speed_y = train_speed_y[-1:,:,:]
# train_speed_x1 = train_speed_x1[:-1,:,:,:]
# train_speed_x2 = train_speed_x2[:-1,:,:,:]
# train_speed_y = train_speed_y[:-1,:,:]


# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed1,train_speed1, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed2,train_speed2, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed3,train_speed3, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed4,train_speed4, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed5,train_speed5, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
# train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed6,train_speed6, look_back, look_back_days, mode)
# train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
# train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
# train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

# print('look_back = ',look_back)
# print('look_back_days = ',look_back_days)
# print('mode = ',mode)
# print('train_speed_x1.shape = ',train_speed_x1.shape)
# print('train_speed_x2.shape = ',train_speed_x2.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x1.shape = ',test_speed_x1.shape)
# print('test_speed_x2.shape = ',test_speed_x2.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[ ]:

# batch_size = train_speed_x.shape[1]

# todaySequence = Input(shape=(look_back, train_speed_x1.shape[3]),name='todaySequence')
# h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False, return_sequences=True)(todaySequence)
# h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False)(h1)

# historySequence = Input(shape=(look_back_days, train_speed_x2.shape[3]),name='historySequence')
# h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False, return_sequences=True)(historySequence)
# h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False)(h2)

# h3 = keras.layers.concatenate([h1, h2])
# predictedSpeed = Dense(1,name='predictedSpeed')(h3)

# model = Model(inputs=[todaySequence, historySequence], outputs=[predictedSpeed])

# model.compile(loss='mean_squared_error', optimizer='adam')

# # model.compile(optimizer='rmsprop',
# #               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
# #               loss_weights={'main_output': 1., 'aux_output': 0.2})

# train_x1 = np.reshape(train_speed_x1,(train_speed_x1.shape[0]*train_speed_x1.shape[1],train_speed_x1.shape[2],train_speed_x1.shape[3]))
# train_x2 = np.reshape(train_speed_x2,(train_speed_x2.shape[0]*train_speed_x2.shape[1],train_speed_x2.shape[2],train_speed_x2.shape[3]))
# train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))

# history = model.fit({'todaySequence': train_x1, 'historySequence': train_x2},
#           {'predictedSpeed': train_y},
#           epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

# # history_plot_historyAsSecondInput(history)


# In[ ]:

# history_plot_historyAsSecondInput(history,'images/history_exp31.png','images/test_exp31.png')


# ### Experiment4: input: univariate speed; output: univariate delta speed; lookback = 15

# In[228]:

# # Mon, Mon_delta, Mon_Z, mon_mean, mon_std = get_certain_dayofweek(Speed,dayofweek = 0)
# train_speed, train_speed_y, _, mean0, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
# test_speed = train_speed[-1:]
# train_speed = train_speed[:-1]
# test_speed_y = train_speed_y[-1:]
# train_speed_y = train_speed_y[:-1]


# print('train_speed.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
train_speed0,train_speed_y0,_,mean0,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
train_speed1,train_speed_y1,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
train_speed2,train_speed_y2,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
train_speed3,train_speed_y3,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
train_speed4,train_speed_y4,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
train_speed5,train_speed_y5,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
train_speed6,train_speed_y6,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# train_speed0 = train_speed0[0:48,:,:]
print('train_speed0.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[229]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset(train_speed,train_speed_y, look_back, mode)
# test_speed_x,test_speed_y = create_dataset(test_speed,test_speed_y, look_back, mode)
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)

look_back = 15
mode = 'uni'
train_speed_x,train_speed_y = create_dataset(train_speed0,train_speed_y0, look_back, mode)

test_speed_x = train_speed_x[-1:,:,:,:]
test_speed_y = train_speed_y[-1:,:,:]
train_speed_x = train_speed_x[:-1,:,:,:]
train_speed_y = train_speed_y[:-1,:,:]

train_speed_x10,train_speed_y0 = create_dataset(train_speed1,train_speed_y1, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed2,train_speed_y2, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed3,train_speed_y3, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed4,train_speed_y4, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed5,train_speed_y5, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed6,train_speed_y6, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[213]:

# plt.plot(test_speed_y[0,390:510,:]+mean0[390:510,0:1])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[230]:

batch_size = train_speed_x.shape[1]

model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# history_plot(history,mean0)


# In[186]:

history_plot(history,'images/history_exp4.png','images/test_exp4.png',a=mean0,b=mean0)
history_plot(history,'images/history_exp4_delta.png','images/test_exp4_delta.png',b=mean0)


# ### Experiment5: input: univariate speed; output: univariate delta speed; lookback = 15; lookback weeks = 15 (as feature);

# In[ ]:




# In[ ]:




# In[ ]:




# ### Experiment6: input: univariate speed; output: univariate delta speed; lookback = 15; lookback weeks = 6 (parallel structure)

# In[187]:

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
# dayofweek = data['dayofweek'][333]
train_speed0,train_speed_y0,_,mean0,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
train_speed1,train_speed_y1,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
train_speed2,train_speed_y2,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
train_speed3,train_speed_y3,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
train_speed4,train_speed_y4,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
train_speed5,train_speed_y5,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
train_speed6,train_speed_y6,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# train_speed0 = train_speed0[0:48,:,:]
print('train_speed0.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[188]:

look_back = 15
look_back_days = 6
mode = 'uni'
train_speed_x1,train_speed_x2,train_speed_y = create_dataset_historyAsSecondInput(train_speed0,train_speed_y0, look_back, look_back_days, mode)

test_speed_x1 = train_speed_x1[-1:,:,:,:]
test_speed_x2 = train_speed_x2[-1:,:,:,:]
test_speed_y = train_speed_y[-1:,:,:]
train_speed_x1 = train_speed_x1[:-1,:,:,:]
train_speed_x2 = train_speed_x2[:-1,:,:,:]
train_speed_y = train_speed_y[:-1,:,:]


train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed1,train_speed_y1, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed2,train_speed_y2, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed3,train_speed_y3, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed4,train_speed_y4, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed5,train_speed_y5, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed6,train_speed_y6, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

print('look_back = ',look_back)
print('look_back_days = ',look_back_days)
print('mode = ',mode)
print('train_speed_x1.shape = ',train_speed_x1.shape)
print('train_speed_x2.shape = ',train_speed_x2.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x1.shape = ',test_speed_x1.shape)
print('test_speed_x2.shape = ',test_speed_x2.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[189]:

# plt.plot(test_speed_y[0,390:510,:]+mean0[390:510,0:1])  #12-19-2016 Monday 6:30AM - 8:30AM


# In[190]:


batch_size = train_speed_x.shape[1]

todaySequence = Input(shape=(look_back, train_speed_x1.shape[3]),name='todaySequence')
# h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False, return_sequences=True)(todaySequence)
h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False,name='h1')(todaySequence)
h1=BatchNormalization()(h1)

historySequence = Input(shape=(look_back_days, train_speed_x2.shape[3]),name='historySequence')
# h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False, return_sequences=True)(historySequence)
h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False,name='h2')(historySequence)
h2=BatchNormalization()(h2)

h3 = keras.layers.concatenate([h1, h2],name='h3')
# h3 = keras.layers.concatenate([h1, h2],name='h3')
h3 = Dropout(0.3)(h3)
predictedSpeed = Dense(1,name='predictedSpeed')(h3)

model = Model(inputs=[todaySequence, historySequence], outputs=[predictedSpeed])

model.compile(loss='mean_squared_error', optimizer='adam')

# model.compile(optimizer='rmsprop',
#               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
#               loss_weights={'main_output': 1., 'aux_output': 0.2})

train_x1 = np.reshape(train_speed_x1,(train_speed_x1.shape[0]*train_speed_x1.shape[1],train_speed_x1.shape[2],train_speed_x1.shape[3]))
train_x2 = np.reshape(train_speed_x2,(train_speed_x2.shape[0]*train_speed_x2.shape[1],train_speed_x2.shape[2],train_speed_x2.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))

history = model.fit({'todaySequence': train_x1, 'historySequence': train_x2},
          {'predictedSpeed': train_y},
          epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

# history_plot_historyAsSecondInput(history,mean0)


# In[ ]:

# intermediate_layer_model_h1 = Model(inputs=model.input,
#                                  outputs=model.get_layer('h1').output)
# intermediate_output_h1 = intermediate_layer_model_h1.predict([train_x1[:1],train_x2[:1]])
# intermediate_layer_model_h2 = Model(inputs=model.input,
#                                  outputs=model.get_layer('h2').output)
# intermediate_output_h2 = intermediate_layer_model_h2.predict([train_x1[:1],train_x2[:1]])
# intermediate_layer_model_h3 = Model(inputs=model.input,
#                                  outputs=model.get_layer('h3').output)
# intermediate_output_h3 = intermediate_layer_model_h3.predict([train_x1[:1],train_x2[:1]])


# In[193]:

history_plot_historyAsSecondInput(history,'images/history_exp6.png','images/test_exp6.png',a=mean0,b=mean0)
history_plot_historyAsSecondInput(history,'images/history_exp6_delta.png','images/test_exp6_delta.png',b=mean0)


# In[239]:

def history_plot_multi(history_object,image1='image1',image2='image2',image3='image3',a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1))):
    
    trainScore = [];
    for i in range(len(train_speed_x)):
        trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x)):
        testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x[0,start,:,:]
    predictions = np.zeros((look_ahead,15))

    for i in range(look_ahead):
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    
    plt.plot(np.arange(look_ahead),predictions[:,:1]+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    predictions_1 = np.zeros((look_ahead,15))

    for i in range(look_ahead):
        trainPredict = test_speed_x[0,start+i,:,:]
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions_1[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions_1[:,:1]+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

    fig3 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('test data', fontsize=20)
    plt.pcolor(test_speed_y[0,start:(start+look_ahead),:].transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('prediction at the start', fontsize=20)
    plt.pcolor(predictions.transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    ax3 = plt.subplot(3,1,3)
    ax3.set_title('prediction using real-time data', fontsize=20)
    plt.pcolor(predictions_1.transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    fig3.savefig(image3, bbox_inches='tight')

def history_plot_multi_historyAsSecondInput(history_object,image1='image1',image2='image2',image3='image3',a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1))):
  
    trainScore = [];
    for i in range(len(train_speed_x1)):
        trainScore.append(model.evaluate([train_speed_x1[i,:,:,:],train_speed_x2[i,:,:,:]], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x1)):
        testScore.append(model.evaluate([test_speed_x1[i,:,:,:],test_speed_x2[i,:,:,:]], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x1[0,start,:,:]  
    predictions = np.zeros((look_ahead,15))

    for i in range(look_ahead):
        input2 = test_speed_x2[0,start+i,:,:]
        prediction = model.predict([np.array([trainPredict]),np.array([input2])], batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions[:,:1]+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    predictions_1 = np.zeros((look_ahead,15))

    for i in range(look_ahead):
        trainPredict = test_speed_x1[0,start+i,:,:]
        input2 = test_speed_x2[0,start+i,:,:]
        prediction = model.predict([np.array([trainPredict]),np.array([input2])], batch_size=batch_size)
        predictions_1[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions_1[:,:1]+a[start:(start+look_ahead),:1],'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),:1]+a[start:(start+look_ahead),:1],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

    fig3 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('test data', fontsize=20)
    plt.pcolor(test_speed_y[0,start:(start+look_ahead),:].transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('prediction at the start', fontsize=20)
    plt.pcolor(predictions.transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    ax3 = plt.subplot(3,1,3)
    ax3.set_title('prediction using real-time data', fontsize=20)
    plt.pcolor(predictions_1.transpose()+a[start:(start+look_ahead),:].transpose(),cmap=my_cmap, vmin=20, vmax=70)
    
    fig3.savefig(image3, bbox_inches='tight')


# ### Experiment7: input: multivariate speed; output: multivariate speed; lookback = 15

# In[246]:

test_speed = Speed[333:334,:,:]
train_speed = Speed[:333,:,:]
print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)


# In[247]:

look_back = 15
mode = 'multi'
train_speed_x,train_speed_y = create_dataset(train_speed,train_speed, look_back, mode)
test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)
print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[248]:

# %matplotlib inline
# plt.plot(test_speed_y[0,390:510,:1])  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(test_speed_y[0,390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM


# In[249]:

batch_size = train_speed_x.shape[1]

model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(Dense(train_speed_y.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
history = model.fit(train_x, train_y,epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# history_plot(history)


# In[250]:

history_plot_multi(history,'images/history_exp7.png','images/test_exp7.png','images/heatmap_exp7.png')


# ### Experiment8: input: multivariate speed; output: multivariate speed; lookback = 15; lookback weeks = 15 (as feature);

# In[ ]:




# In[ ]:




# In[ ]:




# ### Experiment9: input: multivariate speed; output: multivariate speed; lookback = 15; lookback weeks = 6 (parallel structure)

# In[201]:

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
# dayofweek = data['dayofweek'][333]
train_speed0,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
train_speed1,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
train_speed2,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
train_speed3,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
train_speed4,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
train_speed5,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
train_speed6,_,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# train_speed0 = train_speed0[0:48,:,:]
print('train_speed0.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[202]:

look_back = 15
look_back_days = 6
mode = 'multi'
train_speed_x1,train_speed_x2,train_speed_y = create_dataset_historyAsSecondInput(train_speed0,train_speed0, look_back, look_back_days, mode)

test_speed_x1 = train_speed_x1[-1:,:,:,:]
test_speed_x2 = train_speed_x2[-1:,:,:,:]
test_speed_y = train_speed_y[-1:,:,:]
train_speed_x1 = train_speed_x1[:-1,:,:,:]
train_speed_x2 = train_speed_x2[:-1,:,:,:]
train_speed_y = train_speed_y[:-1,:,:]


train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed1,train_speed1, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed2,train_speed2, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed3,train_speed3, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed4,train_speed4, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed5,train_speed5, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed6,train_speed6, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

print('look_back = ',look_back)
print('look_back_days = ',look_back_days)
print('mode = ',mode)
print('train_speed_x1.shape = ',train_speed_x1.shape)
print('train_speed_x2.shape = ',train_speed_x2.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x1.shape = ',test_speed_x1.shape)
print('test_speed_x2.shape = ',test_speed_x2.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[99]:

# plt.plot(test_speed_y[0,390:510,:1])  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(test_speed_y[0,390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM


# In[203]:

batch_size = train_speed_x.shape[1]

todaySequence = Input(shape=(look_back, train_speed_x1.shape[3]),name='todaySequence')
#h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False, return_sequences=True)(todaySequence)
h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False)(todaySequence)
h1 = BatchNormalization()(h1)

historySequence = Input(shape=(look_back_days, train_speed_x2.shape[3]),name='historySequence')
#h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False, return_sequences=True)(historySequence)
h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False)(historySequence)
h2 = BatchNormalization()(h2)

h3 = keras.layers.concatenate([h1, h2])
h3 = Dropout(0.3)(h3)
predictedSpeed = Dense(train_speed_y.shape[2],name='predictedSpeed')(h3)

model = Model(inputs=[todaySequence, historySequence], outputs=[predictedSpeed])

model.compile(loss='mean_squared_error', optimizer='adam')

# model.compile(optimizer='rmsprop',
#               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
#               loss_weights={'main_output': 1., 'aux_output': 0.2})

train_x1 = np.reshape(train_speed_x1,(train_speed_x1.shape[0]*train_speed_x1.shape[1],train_speed_x1.shape[2],train_speed_x1.shape[3]))
train_x2 = np.reshape(train_speed_x2,(train_speed_x2.shape[0]*train_speed_x2.shape[1],train_speed_x2.shape[2],train_speed_x2.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))

history = model.fit({'todaySequence': train_x1, 'historySequence': train_x2},
          {'predictedSpeed': train_y},
          epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

# history_plot_historyAsSecondInput(history)


# In[205]:

history_plot_multi_historyAsSecondInput(history,'images/history_exp9.png','images/test_exp9.png','images/heatmap_exp9.png')


# ### Experiment10: input: multivariate speed; output: multivariate delta speed; lookback = 15

# In[231]:

# # Mon, Mon_delta, Mon_Z, mon_mean, mon_std = get_certain_dayofweek(Speed,dayofweek = 0)
# train_speed, train_speed_y, _, mean0, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
# test_speed = train_speed[-1:]
# train_speed = train_speed[:-1]
# test_speed_y = train_speed_y[-1:]
# train_speed_y = train_speed_y[:-1]


# print('train_speed.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
train_speed0,train_speed_y0,_,mean0,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
train_speed1,train_speed_y1,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
train_speed2,train_speed_y2,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
train_speed3,train_speed_y3,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
train_speed4,train_speed_y4,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
train_speed5,train_speed_y5,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
train_speed6,train_speed_y6,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# train_speed0 = train_speed0[0:48,:,:]
print('train_speed0.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[232]:

# look_back = 15
# mode = 'uni'
# train_speed_x,train_speed_y = create_dataset(train_speed,train_speed_y, look_back, mode)
# test_speed_x,test_speed_y = create_dataset(test_speed,test_speed_y, look_back, mode)
# print('look_back = ',look_back)
# print('mode = ',mode)
# print('train_speed_x.shape = ',train_speed_x.shape)
# print('train_speed_y.shape = ',train_speed_y.shape)
# print('test_speed_x.shape = ',test_speed_x.shape)
# print('test_speed_y.shape = ',test_speed_y.shape)

look_back = 15
mode = 'multi'
train_speed_x,train_speed_y = create_dataset(train_speed0,train_speed_y0, look_back, mode)

test_speed_x = train_speed_x[-1:,:,:,:]
test_speed_y = train_speed_y[-1:,:,:]
train_speed_x = train_speed_x[:-1,:,:,:]
train_speed_y = train_speed_y[:-1,:,:]

train_speed_x10,train_speed_y0 = create_dataset(train_speed1,train_speed_y1, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed2,train_speed_y2, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed3,train_speed_y3, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed4,train_speed_y4, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed5,train_speed_y5, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_y0 = create_dataset(train_speed6,train_speed_y6, look_back,mode)
train_speed_x = np.concatenate((train_speed_x,train_speed_x10),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

print('look_back = ',look_back)
print('look_back_days = ',look_back_days)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[208]:

# plt.plot(test_speed_y[0,390:510,:1]+mean0[390:510,:1])  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(mean0[390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(test_speed_y[0,390:510,:].transpose()+mean0[390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM


# In[237]:

batch_size = train_speed_x.shape[1]

model = Sequential()
# model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(train_speed_y.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
# history_plot(history,mean0)


# In[242]:

history_plot_multi(history,'images/history_exp10.png','images/test_exp10.png','images/heatmap_exp10.png',a=mean0,b=mean0)
history_plot_multi(history,'images/history_exp10_delta.png','images/test_exp10_delta.png','images/heatmap_exp10_delta.png',b=mean0)


# ### Experiment11: input: multivariate speed; output: multivariate delta speed; lookback = 15; lookback weeks = 15 (as feature);

# In[ ]:




# In[ ]:




# In[ ]:




# ### Experiment12: input: multivariate speed; output: multivariate delta speed; lookback = 15; lookback weeks = 6 (parallel structure)

# In[243]:

# test_speed = Speed[333:334,:,:]
# train_speed = Speed[:334,:,:]
# dayofweek = data['dayofweek'][333]
train_speed0,train_speed_y0,_,mean0,_ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
train_speed1,train_speed_y1,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 1)
train_speed2,train_speed_y2,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 2)
train_speed3,train_speed_y3,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 3)
train_speed4,train_speed_y4,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 4)
train_speed5,train_speed_y5,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 5)
train_speed6,train_speed_y6,_,_,_ = get_certain_dayofweek(Speed[:334],dayofweek = 6)
# train_speed0 = train_speed0[0:48,:,:]
print('train_speed0.shape = ',train_speed.shape)
# print('test_speed.shape = ',test_speed.shape)


# In[244]:

look_back = 15
look_back_days = 6
mode = 'multi'
train_speed_x1,train_speed_x2,train_speed_y = create_dataset_historyAsSecondInput(train_speed0,train_speed_y0, look_back, look_back_days, mode)

test_speed_x1 = train_speed_x1[-1:,:,:,:]
test_speed_x2 = train_speed_x2[-1:,:,:,:]
test_speed_y = train_speed_y[-1:,:,:]
train_speed_x1 = train_speed_x1[:-1,:,:,:]
train_speed_x2 = train_speed_x2[:-1,:,:,:]
train_speed_y = train_speed_y[:-1,:,:]


train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed1,train_speed_y1, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed2,train_speed_y2, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed3,train_speed_y3, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed4,train_speed_y4, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed5,train_speed_y5, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)
train_speed_x10,train_speed_x20,train_speed_y0 = create_dataset_historyAsSecondInput(train_speed6,train_speed_y6, look_back, look_back_days, mode)
train_speed_x1 = np.concatenate((train_speed_x1,train_speed_x10),axis=0)
train_speed_x2 = np.concatenate((train_speed_x2,train_speed_x20),axis=0)
train_speed_y = np.concatenate((train_speed_y,train_speed_y0),axis=0)

print('look_back = ',look_back)
print('look_back_days = ',look_back_days)
print('mode = ',mode)
print('train_speed_x1.shape = ',train_speed_x1.shape)
print('train_speed_x2.shape = ',train_speed_x2.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x1.shape = ',test_speed_x1.shape)
print('test_speed_x2.shape = ',test_speed_x2.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[122]:

# plt.plot(test_speed_y[0,390:510,:1]+mean0[390:510,:1])  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(mean0[390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM
# plt.pcolor(test_speed_y[0,390:510,:].transpose()+mean0[390:510,:].transpose(),cmap=my_cmap, vmin=20, vmax=70)  #12-19-2016 Monday 6:30AM - 8:30AM


# In[123]:


batch_size = train_speed_x.shape[1]

todaySequence = Input(shape=(look_back, train_speed_x1.shape[3]),name='todaySequence')
#h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False, return_sequences=True)(todaySequence)

h1=LSTM(32, input_shape=(look_back, train_speed_x1.shape[3]), stateful=False,name='h1')(todaySequence)
h1 = BatchNormalization()(h1)

historySequence = Input(shape=(look_back_days, train_speed_x2.shape[3]),name='historySequence')
#h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False, return_sequences=True)(historySequence)
h2=LSTM(32, input_shape=(look_back, train_speed_x2.shape[3]), stateful=False,name='h2')(historySequence)
h2 = BatchNormalization()(h2)

h3 = keras.layers.concatenate([h1, h2],name='h3')
h3 = Dropout(0.3)(h3)
# h3 = keras.layers.concatenate([h1, h2],name='h3')

predictedSpeed = Dense(train_speed_y.shape[2],name='predictedSpeed')(h3)

model = Model(inputs=[todaySequence, historySequence], outputs=[predictedSpeed])

model.compile(loss='mean_squared_error', optimizer='adam')

# model.compile(optimizer='rmsprop',
#               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
#               loss_weights={'main_output': 1., 'aux_output': 0.2})

train_x1 = np.reshape(train_speed_x1,(train_speed_x1.shape[0]*train_speed_x1.shape[1],train_speed_x1.shape[2],train_speed_x1.shape[3]))
train_x2 = np.reshape(train_speed_x2,(train_speed_x2.shape[0]*train_speed_x2.shape[1],train_speed_x2.shape[2],train_speed_x2.shape[3]))
train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))

history = model.fit({'todaySequence': train_x1, 'historySequence': train_x2},
          {'predictedSpeed': train_y},
          epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

# history_plot_historyAsSecondInput(history,mean0)


# In[126]:

history_plot_multi_historyAsSecondInput(history,'images/history_exp12.png','images/test_exp12.png','images/heatmap_exp12.png',a=mean0,b=mean0)
history_plot_multi_historyAsSecondInput(history,'images/history_exp12_delta.png','images/test_exp12_delta.png','images/heatmap_exp12_delta.png',b=mean0)


# In[245]:
