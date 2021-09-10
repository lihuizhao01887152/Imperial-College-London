#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Folder name in which results are saved
name = "ANN_file"

os.chdir('C:\\Users\\LIHUI ZHAO')

if not os.path.exists(name):
    os.mkdir(name)
os.chdir(name)

#!/usr/bin/env python

'''ANN.py'''

__author__     = 'LIHUI ZHAO'
__email__      = 'lz620@imperial.ac.uk'

## 2021/08/31

[globals().pop(var) for var in dir() if not var.startswith("__")]

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import pandas as pd
import keras
from keras import metrics
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers, Input, callbacks, Model, utils, regularizers
import time
import matplotlib.pyplot as plt
import sklearn
import math
import os 

'''
Block 1 Loading Data
'''

name1 = "R2 VS training ratio"

os.chdir('C:\\Users\\LIHUI ZHAO\\ANN_file')

if not os.path.exists(name1):
    os.mkdir(name1)
os.chdir(name1)


get_ipython().run_line_magic('matplotlib', '')


# store R2 results list
li_for_graph=[]
li_for_graph_2=[]

# training ratio list
li_for_loop = [element/10 for element in [i for i in range(1,11,1)]]



for training_ratio in li_for_loop:
    print('\n Training ratio = %s'%training_ratio)
    start = time.time()    
    # Data Loading (DataFrame)
    df=pd.read_csv('C:\\Users\\LIHUI ZHAO\\Joule-Thomson2.csv') 
    # DataFrame convert to numpy.ndarray
    d = df.values   
    # convert to float32
    data = np.float32(d)                                     
    # Input and Output split
    x_data, y_data = np.split(data,[2],1)                     
    x_data_mean = np.mean(x_data,axis=0)
    x_data_std =np.std(x_data,axis=0)
    x_data_max =np.max(x_data,axis=0)
    x_data_min =np.min(x_data,axis=0)
    x_data = (x_data - x_data_mean)/x_data_std             # Z-score scaling
    #x_data =(x_data - x_data_min)/(x_data_max - x_data_min) # min_max scaling  #sklearn preprocessing--scaling--power transform (log)
    # Horizontal stack x_data和y_data
    data =np.hstack((x_data,y_data))
    # Convert to DataFrame
    df = pd.DataFrame(data)            
    # DataFrame train test split 
    train_set = df.sample(frac=0.8,random_state=0) 
    test_set = df.drop(train_set.index)    
    #rest_set = df.drop(train_set.index)
    #test_set = rest_set.sample(frac=0.5, random_state=0)
    #validation_set = rest_set.drop(test_set.index)
    # DataFrame convert to numpy.ndarray, then convert to float32
    train_set =train_set.values
    train_set = np.float32(train_set)
    test_set =test_set.values
    test_set = np.float32(test_set)
    # Input and Output split
    x_data_train, y_data_train = np.split(train_set,[2],1)
    x_data_test, y_data_test = np.split(test_set,[2],1)
    # Name convert
    x_train = x_data_train
    y_train = y_data_train
    x_test = x_data_test
    y_test = y_data_test
    print('x_train.shape:',x_train.shape)
    print('y_train.shape:',y_train.shape)
    print('x_test.shape:',x_test.shape)
    print('y_test.shape:',y_test.shape)

    '''
    Block 2 Building ANN
    '''

    #regularizer
    regularizer = regularizers.l1_l2(0, 0)

    lr_setting = 0.2

    # Build ANN Sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10, input_dim = 2, activation = tf.nn.tanh, 
                                    kernel_regularizer = regularizer,kernel_initializer='glorot_uniform', bias_initializer='zeros')) ###batch_input_shape=(None,2)
    model.add(tf.keras.layers.Dense(units = 6, activation=tf.nn.tanh))   
    model.add(tf.keras.layers.Dense(units = 3, activation=tf.nn.tanh))
    model.add(tf.keras.layers.Dense(units = 1, activation=None))   
    # Optimizer
    optimizer1 = tf.optimizers.SGD(learning_rate=lr_setting)    
    # Loss Function
    model.compile(optimizer=optimizer1,
                    loss='mse', metrics = ['mae'], ) 

    '''
    Block 3 Training ANN
    '''

    # Number of epochs
    epoch_num = 200   

    print('Start training for {} epochs ...'.format(epoch_num))

    # Store Train_loss and Test_loss for plot
    fig_loss_training = np.zeros([epoch_num])  
    fig_loss_testing = np.zeros([epoch_num])

    # Store metrics —— mae
    fig_mae_training = np.zeros([epoch_num])  
    fig_mae_testing = np.zeros([epoch_num]) 


    print('\n lr equals {}'.format(lr_setting))

    # Learning Rate scheduler
    def scheduler(epoch):

        # test loss for each epoch
        if epoch % 1 ==0:
            print('.', end=' ')
            # produce test_cost
            test_cost = model.evaluate(x_test, y_test, batch_size=345, verbose=0) 
            count = epoch - 1
            # store test_cost and mae
            fig_loss_testing[count] += test_cost[0]
            fig_mae_testing[count] +=test_cost[1]

        # Learning rate decrease for epochs 
        if epoch % (epoch_num//4) == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr *0.5)
            print("\n lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    # Produce train_cost
    train_cost = model.fit(x_train, y_train, epochs=epoch_num, batch_size=128, shuffle=True, validation_split= 1 - training_ratio, verbose=0, 
                           callbacks=[reduce_lr])  #default batch_size=32

    ###print(train_cost.history)

    # Store train_cost
    for count in range (epoch_num):
        fig_loss_training[count] = train_cost.history['loss'][count]

    # # Store train mae
    for count in range (epoch_num):
        fig_mae_training[count] = train_cost.history['mae'][count]


    from sklearn.metrics import mean_squared_error 
    from sklearn.metrics import mean_absolute_error 
    from sklearn.metrics import r2_score
    import numpy as np

    y_predict = model.predict(x_test)
    a=y_test
    b=y_predict

    # Metrics by sklearn
    MSE=mean_squared_error(a,b)
    RMSE=np.sqrt(mean_squared_error(a,b))
    MAE=mean_absolute_error(a,b)
    R2=r2_score(a,b)
    
    y_predict_train =model.predict(x_train)
    c=y_train
    d=y_predict_train    
    R22=r2_score(c,d)
    

    print('\n Test MSE: {}'.format(MSE))
    print('\n Test RMSE: {}'.format(RMSE))
    print('\n Test MAE: {}'.format(MAE))
    print('\n Test R-square: {}'.format(R2))

    end = time.time()
    print('\n Running time: %s Seconds' % (end - start))
    

# store some results in Excel file
    hist2 = pd.DataFrame(fig_loss_testing)
    ###hist2['epoch'] = train_cost.epoch
    hist2.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\R2 VS training ratio\\hist{}.xls'.format(training_ratio))

    li_for_graph.append(R2)
    li_for_graph_2.append(R22)
print(li_for_graph)
print(li_for_graph_2)

# plot 
plt.plot(li_for_loop, li_for_graph)
plt.xlabel('trainning data ratio')
plt.ylabel('R-square')
plt.savefig('R-square VS training ratio.png')
plt.show()


r2_size=pd.DataFrame(li_for_loop)
r2_size['li_for_graph']=li_for_graph
r2_size.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\R2 VS training ratio\\r2_vs_train_size.xls')

