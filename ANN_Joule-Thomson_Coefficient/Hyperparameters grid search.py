#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np
import time
from keras import metrics
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers, Input, callbacks, Model, utils, regularizers
import tensorflow as tf

name3 = "hyperparameter grid search results"
os.chdir('C:\\Users\\LIHUI ZHAO\\ANN_file')
if not os.path.exists(name3):
    os.mkdir(name3)
os.chdir(name3)
start = time.time()
get_ipython().run_line_magic('matplotlib', '')
li_for_graph2 = []
li_for_loop2 =[0.4, 0.2, 0.1, 0.05]
for lr_setting in li_for_loop2:
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
    
    # Horizontal stack x_data和y_data
    data =np.hstack((x_data,y_data))
    # Convert to DataFrame
    df = pd.DataFrame(data)            
    # DataFrame train test split 
    train_set = df.sample(frac=0.8,random_state=0)  
    test_set = df.drop(train_set.index)    
    
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


    '''
    Block 2 Building ANN
    '''

    #regularizer
    regularizer = regularizers.l1_l2(0, 0)

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
    
    bz_setting = 128
    
    print('\n bz equals {}'.format(bz_setting))

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
    train_cost = model.fit(x_train, y_train, epochs=epoch_num, batch_size=bz_setting, shuffle=True, validation_split= 0.2, verbose=0, 
                           callbacks=[reduce_lr])  

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

    end = time.time()
    print('\n Running time: %s Seconds' % (end - start))
    
    li_for_graph2.append(pd.DataFrame(train_cost.history).iloc[epoch_num-1,2])


print(li_for_graph2)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(li_for_loop2, li_for_graph2)
plt.xlabel('learning rate')
plt.ylabel('validation loss')
plt.savefig("grid_search_validation1.png")
plt.close("all")


[globals().pop(var) for var in dir() if not var.startswith("__")]

import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np
import time
from keras import metrics
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers, Input, callbacks, Model, utils, regularizers
import tensorflow as tf

name3 = "hyperparameter grid search results"

os.chdir('C:\\Users\\LIHUI ZHAO\\ANN_file')

if not os.path.exists(name3):
    os.mkdir(name3)
os.chdir(name3)

start = time.time()

# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('matplotlib', '')

li_for_graph2 = []
li_for_loop2 = [50, 100, 200, 400]

for epoch_num in li_for_loop2:
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
    
    # Horizontal stack x_data和y_data
    data =np.hstack((x_data,y_data))
    # Convert to DataFrame
    df = pd.DataFrame(data)           
    # DataFrame train test split 
    train_set = df.sample(frac=0.8,random_state=0) 
    test_set = df.drop(train_set.index)    

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


    '''
    Block 2 Building ANN
    '''

    #regularizer
    regularizer = regularizers.l1_l2(0, 0)
    #keras.regularizers.l1(0.)
    #keras.regularizers.l2(0.)

    # weight bias initial

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


    print('Start training for {} epochs ...'.format(epoch_num))

    # Store Train_loss and Test_loss for plot
    fig_loss_training = np.zeros([epoch_num])  
    fig_loss_testing = np.zeros([epoch_num])

    # Store metrics —— mae
    fig_mae_training = np.zeros([epoch_num])  
    fig_mae_testing = np.zeros([epoch_num]) 


    print('\n lr equals {}'.format(lr_setting))
    
    bz_setting = 128
    
    print('\n bz equals {}'.format(bz_setting))

    # Learning Rate scheduler
    def scheduler(epoch):

        # test loss for each epoch
        if epoch % 1 ==0:
            print('.', end=' ')
            # produce test_cost
            test_cost = model.evaluate(x_test, y_test, batch_size=345, verbose=0) 
            count = epoch - 1
            # store test_cost and mae
            ###print('test_cost', test_cost[0])
            fig_loss_testing[count] += test_cost[0]
            ###print('test mae', test_cost[1])
            fig_mae_testing[count] +=test_cost[1]

        # Learning rate decrease for epochs 
        if epoch % (epoch_num//4) == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr *0.5)
            print("\n lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
    
    bz_setting = 128
    # Produce train_cost
    train_cost = model.fit(x_train, y_train, epochs=epoch_num, batch_size=bz_setting, shuffle=True, validation_split= 0.2, verbose=0, 
                           callbacks=[reduce_lr])  

    ###print(train_cost.history)

    # Store train_cost
    for count in range (epoch_num):
        ###print('train_cost',train_cost.history['loss'][count])
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


    end = time.time()
    print('\n Running time: %s Seconds' % (end - start))
    
    li_for_graph2.append(pd.DataFrame(train_cost.history).iloc[epoch_num-1,2])

print(li_for_graph2)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(li_for_loop2, li_for_graph2)
plt.xlabel('epoch_num')
plt.ylabel('validation loss')
plt.savefig("grid_search_validation2.png")
plt.close("all")

[globals().pop(var) for var in dir() if not var.startswith("__")]

import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np
import time
from keras import metrics
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers, Input, callbacks, Model, utils, regularizers
import tensorflow as tf

name3 = "hyperparameter grid search results"

os.chdir('C:\\Users\\LIHUI ZHAO\\ANN_file')

if not os.path.exists(name3):
    os.mkdir(name3)
os.chdir(name3)

start = time.time()

# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('matplotlib', '')

li_for_graph2 = []
li_for_loop2 =[ 64, 128, 256, 512]

for bz_setting in li_for_loop2:
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


    '''
    Block 2 Building ANN
    '''

    #regularizer
    regularizer = regularizers.l1_l2(0, 0)
    #keras.regularizers.l1(0.)
    #keras.regularizers.l2(0.)

    # weight bias initial

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

    print('lr equals {} ...'.format(lr_setting))
    
    print('bz equals {} ...'.format(bz_setting))

    # Learning Rate scheduler
    def scheduler(epoch):

        # test loss for each epoch
        if epoch % 1 ==0:
            print('.', end=' ')
            # produce test_cost
            test_cost = model.evaluate(x_test, y_test, batch_size=345, verbose=0) 
            count = epoch - 1
            # store test_cost and mae
            ###print('test_cost', test_cost[0])
            fig_loss_testing[count] += test_cost[0]
            ###print('test mae', test_cost[1])
            fig_mae_testing[count] +=test_cost[1]

        # Learning rate decrease for epochs 
        if epoch % (epoch_num//4) == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr *0.5)
            print("\n lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
    
    
    # Produce train_cost
    train_cost = model.fit(x_train, y_train, epochs=epoch_num, batch_size=bz_setting, shuffle=True, validation_split= 0.2, verbose=0, 
                           callbacks=[reduce_lr])  

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


    end = time.time()
    print('\n Running time: %s Seconds' % (end - start))
    
    li_for_graph2.append(pd.DataFrame(train_cost.history).iloc[epoch_num-1,2])

print(li_for_graph2)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(li_for_loop2, li_for_graph2)
plt.xlabel('batch size')
plt.ylabel('validation loss')
plt.savefig("grid_search_validation3.png")
plt.close("all")

