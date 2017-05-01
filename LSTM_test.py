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
import numpy as np
import csv
from sklearn.utils import shuffle
from pylab import *
from matplotlib import gridspec

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



test_speed = Speed[-1:,:,:]
dayofweek = data['dayofweek'][len(data)-1]
train_speed = Speed[data.index[data['dayofweek'] == dayofweek],:,:]
print('train_speed.shape = ',train_speed.shape)
print('test_speed.shape = ',test_speed.shape)


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


look_ahead = 60
start = 0
trainPredict = test_speed_x[0,start,:,:]
predictions = np.zeros((look_ahead,1))

for i in range(look_ahead):
    prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
    predictions[i] = prediction
    trainPredict = np.vstack([trainPredict[1:],prediction])
    
fig = plt.figure(figsize=(12,10))
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

fig.savefig('test_output.png')

