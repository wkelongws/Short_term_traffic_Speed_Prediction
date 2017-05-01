
# coding: utf-8

# In[ ]:

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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


# In[ ]:

import pickle
(Traffic,Speed,data) = pickle.load( open( "speed_short_term.p", "rb" ) )
print(Traffic.shape)
print(Speed.shape)
data.tail(5)


# In[ ]:

def shapeback(Y):
    YY = np.reshape(Y[len(Y)%288:,:],(len(Y)//288,288,Y.shape[1]))
    return np.swapaxes(YY,1,2)

def create_dataset(Speed, look_back=6, mode='uni'):
    
    dataX,dataY = [],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,0:1]

            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset[i + look_back, :])
                
            dataX.append(numpy.array(dataX_))
            dataY.append(numpy.array(dataY_))
            
    if mode == 'multi':
        
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,:]

            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset[i + look_back, :])
                
            dataX.append(numpy.array(dataX_))
            dataY.append(numpy.array(dataY_))    
                
                
    return numpy.array(dataX), numpy.array(dataY)


# ### Experiment1: speed to speed univariate, same day of week, stateful

# In[ ]:

test_speed = Speed[-1:,:,:]
dayofweek = data['dayofweek'][len(data)-1]
train_speed = Speed[data.index[data['dayofweek'] == dayofweek],:,:]
print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)


# In[ ]:

look_back = 15
mode = 'uni'
train_speed_x,train_speed_y = create_dataset(train_speed, look_back, mode)
test_speed_x,test_speed_y = create_dataset(test_speed, look_back, mode)
print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

# plt.plot(test_speed_y[0,:,:])


# In[ ]:

batch_size = 1
epochs = 1
model = Sequential()
model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for e in range(epochs):
    for i in range(len(train_speed_x)):#len(train_speed_x)
        model.fit(train_speed_x[i,:,:,:], train_speed_y[i,:,:], nb_epoch=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()


# In[ ]:

# trainScore = [];
# for i in range(len(train_speed_x)):
#     trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
# trainScore = np.mean(trainScore)
# print('Train Score: ', trainScore)
# testScore = [];
# for i in range(len(test_speed_x)):
#     testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
# testScore = np.mean(testScore)
# print('Test Score: ', testScore)


# In[ ]:

# batch_size = 1

look_ahead = 60
start = 0
trainPredict = test_speed_x[0,start,:,:]
predictions = np.zeros((look_ahead,1))

for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
    predictions[i] = prediction
    trainPredict = np.vstack([trainPredict[1:],prediction])
    
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

fig1.savefig('test_output_exp1.png')


# In[ ]:




# ### Experiment2: speed to speed univariate, same day of week, stateless

# In[ ]:

test_speed = Speed[-1:,:,:]
dayofweek = data['dayofweek'][len(data)-1]
train_speed = Speed[data.index[data['dayofweek'] == dayofweek],:,:]
train_speed = train_speed[:-1,:,:]
print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)


# In[ ]:

look_back = 15
mode = 'uni'
train_speed_x,train_speed_y = create_dataset(train_speed, look_back, mode)
test_speed_x,test_speed_y = create_dataset(test_speed, look_back, mode)
print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

batch_size = 1024
epochs = 2
model = Sequential()
model.add(LSTM(32, input_shape=(look_back, 1), stateful=False, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, 1), stateful=False))
# model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = np.reshape(train_speed_x,(49*1424,15,1))
train_y = np.reshape(train_speed_y,(49*1424,1))
h2 = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, verbose=1, shuffle=True)



# In[ ]:

### plot the training and validation loss for each epoch
fig2 = plt.figure(figsize=(15,5))
plt.plot(h2.history['loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')

fig2.savefig('train_history_exp2.png', bbox_inches='tight')


# In[ ]:

# batch_size = 1

look_ahead = 60
start = 0
trainPredict = test_speed_x[0,start,:,:]
predictions = np.zeros((look_ahead,1))

for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
    predictions[i] = prediction
    trainPredict = np.vstack([trainPredict[1:],prediction])
    
fig3 = plt.figure(figsize=(12,10))
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

fig3.savefig('test_output_exp2.png')


# ### Experiment3: speed to speed univariate, last month, stateful

# In[ ]:




# ### Experiment4: speed to speed univariate, last month, stateless

# In[ ]:

test_speed = Speed[-1:,:,:]
# dayofweek = data['dayofweek'][len(data)-1]
# train_speed = Speed[data.index[data['dayofweek'] == dayofweek],:,:]
train_speed = Speed[:-1,:,:]
print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)


# In[ ]:

look_back = 15
mode = 'uni'
train_speed_x,train_speed_y = create_dataset(train_speed, look_back, mode)
test_speed_x,test_speed_y = create_dataset(test_speed, look_back, mode)
print('look_back = ',look_back)
print('mode = ',mode)
print('train_speed_x.shape = ',train_speed_x.shape)
print('train_speed_y.shape = ',train_speed_y.shape)
print('test_speed_x.shape = ',test_speed_x.shape)
print('test_speed_y.shape = ',test_speed_y.shape)


# In[ ]:

batch_size = 1024
epochs = 1
model = Sequential()
model.add(LSTM(32, input_shape=(look_back, 1), stateful=False, return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(32, input_shape=(look_back, 1), stateful=False))
# model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = np.reshape(train_speed_x,(345*1424,15,1))
train_y = np.reshape(train_speed_y,(345*1424,1))
h4 = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, verbose=1, shuffle=True)



# In[ ]:

### plot the training and validation loss for each epoch
fig4 = plt.figure(figsize=(15,5))
plt.plot(h2.history['loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')

fig4.savefig('train_history_exp4.png', bbox_inches='tight')


# In[ ]:

# batch_size = 1

look_ahead = 60
start = 0
trainPredict = test_speed_x[0,start,:,:]
predictions = np.zeros((look_ahead,1))

for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
    predictions[i] = prediction
    trainPredict = np.vstack([trainPredict[1:],prediction])
    
fig5 = plt.figure(figsize=(12,10))
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

fig5.savefig('test_output_exp4.png')


# ### Experiment5: speed to speed multivariate, best of above

# In[ ]:




# ### Experiment6: speed, volumn, occup to speed univariate, best of above

# In[ ]:




# ### Experiment6: speed, volumn, occup to speed multivariate, best of above

# In[ ]:



