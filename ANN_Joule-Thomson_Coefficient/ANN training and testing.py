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



name2 = "Final_model_results"

os.chdir('C:\\Users\\LIHUI ZHAO\\ANN_file')

if not os.path.exists(name2):
    os.mkdir(name2)
os.chdir(name2)

start = time.time()

# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('matplotlib', '')
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
print('x_train.shape:',x_train.shape)
print('y_train.shape:',y_train.shape)
print('x_test.shape:',x_test.shape)
print('y_test.shape:',y_test.shape)

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
train_cost = model.fit(x_train, y_train, epochs=epoch_num, batch_size=128, shuffle=True, validation_split= 0, verbose=0, 
                       callbacks=[reduce_lr])  

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

print('\n Test MSE: {}'.format(MSE))
print('\n Test RMSE: {}'.format(RMSE))
print('\n Test MAE: {}'.format(MAE))
print('\n Test R-square: {}'.format(R2))

y_train_predict = model.predict(x_train)
c=y_train
d=y_train_predict

MSE_t=mean_squared_error(c,d)
RMSE_t=np.sqrt(mean_squared_error(c,d))
MAE_t=mean_absolute_error(c,d)
R2_t=r2_score(c,d) 

print('\n Test MSE_t: {}'.format(MSE_t))
print('\n Test RMSE_t: {}'.format(RMSE_t))
print('\n Test MAE_t: {}'.format(MAE_t))
print('\n Test R-square_t: {}'.format(R2_t))

end = time.time()
print('\n Running time: %s Seconds' % (end - start))

###hist = pd.DataFrame(train_cost.history)
###hist['epoch'] = train_cost.epoch
###hist.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\hist.xls')

'''
Block 4 Plotting graphs
'''

'''
Block 5 Generating Results Files
'''



# try to find the index of Press from PVT data        

import matplotlib.pyplot as plt


# Scatter JT predict vs test  +  y=x
x=np.linspace(-2,10, 50)
y=x
fig1 = plt.figure()   
ax1 = fig1.add_subplot()
plt.plot(x,y,color='black',linewidth=3, linestyle='--')  
y_predict = model.predict(x_test)
ax1.scatter(y_test, y_predict, color='red')
plt.xlim((-2,8))  
plt.ylim((-2,8))  
plt.xlabel(r'$Test\ Joule\  Thomson\  Coefficient$')  
plt.ylabel(r'$Predict\ Joule\  Thomson\  Coefficient$')
plt.savefig("Scatter JT predict vs test + y=x.png")
plt.show()
plt.close('all')

y_x=pd.DataFrame(y_predict)
y_x['y_test']=y_test
y_x.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\y_x.xls')



# Scatter (test/predict) JT vs T
fig3, ax3 = plt.subplots()                               
ax4 = ax3.twinx()    
lns1 = ax3.scatter(x_test[:,1],y_predict, label="Predict JTC")   
lns2 = ax4.scatter(x_test[:,1],y_test, color='r', label="Test JTC")   
ax3.set_xlabel('T')
ax3.set_ylabel('Predict JTC')
ax4.set_ylabel('Test JTC')
ax3.set_ylim((-1,8))  
ax4.set_ylim((-1,8))
lns3 = [lns1, lns2]   
labels1 = [l.get_label() for l in lns3]
plt.legend(lns3, labels1, loc='best')  
plt.savefig("Scatter test JT predict JT vs T.png")
plt.close('all')

phase_cover = pd.DataFrame(x_test[:,1])
phase_cover['y_predict'] = y_predict
phase_cover['y_test'] = y_test
phase_cover.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\phase_cover.xls')


# Train loss vs test loss
fig4, ax5 = plt.subplots()                               
ax6 = ax5.twinx()    
lns4 = ax5.plot(np.arange(epoch_num), fig_loss_training, label="Loss_train")   
lns5 = ax6.plot(np.arange(epoch_num), fig_loss_testing, linestyle='--', color='r', label="Loss_test")   
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Train loss')
ax6.set_ylabel('Test loss')
ax5.set_ylim((0,2))  
ax6.set_ylim((0,2))
lns6 = lns4 + lns5   
labels2 = ["Loss_train", "Loss_test"]
plt.legend(lns6, labels2, loc='best')   
plt.savefig("Train loss vs test loss.png")
plt.close('all')

loss_mse = pd.DataFrame(np.arange(epoch_num))
loss_mse['fig_loss_training'] =fig_loss_training
loss_mse['fig_loss_testing'] =fig_loss_testing
loss_mse.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\loss_mse.xls')



# MAE
fig5, ax7 = plt.subplots()                              
ax8 = ax7.twinx()   
lns7 = ax7.plot(np.arange(epoch_num), fig_mae_training, label="Train MAE")   
lns8 = ax8.plot(np.arange(epoch_num), fig_mae_testing, linestyle='--', color='r', label="Test MAE")   
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Train MAE')
ax8.set_ylabel('Test MAE')
ax7.set_ylim((0,2))  
ax8.set_ylim((0,2))
lns9 = lns7 + lns8   
labels3 = ["Train MAE", "Test MAE"]
plt.legend(lns9, labels3, loc='best')   
plt.savefig("Train MAE vs Test MAE.png")
plt.close('all')

loss_mae = pd.DataFrame(np.arange(epoch_num))
loss_mae['fig_mae_training'] = fig_mae_training
loss_mae['fig_mae_testing'] =fig_mae_testing
loss_mae.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\loss_mae.xls')

#plt.pause(5)
#plt.clf()  

'''
Block  Reverse norm data
'''
import matplotlib.pyplot as plt
def Reverse_Norm(li, index):
    new_li = []
    for _ in range(len(li)):
        element = li[_] * x_data_std[index] + x_data_mean[index]
        new_li.append(element)
    return new_li

def Norm_fun(li, index):
    new_li = []
    for _ in range(len(li)):
        element = (li[_] - x_data_mean[index])/(x_data_std[index])
        new_li.append(element)
    return new_li

x_test[:, 0] = Reverse_Norm(x_test[:,0], index=0)
x_test[:, 1] = Reverse_Norm(x_test[:,1], index=1)

'''
Block  JTIC FOR RHO AND TEMPERATURE phase diagram
'''
# Scatter real rho vs real T （JT=0)
# Store y_predict value which is between -0.1 to 0.1
rho_li=[]
tem_li=[]
for index in range(345):
    if y_predict[index][0]<0.04 and y_predict[index][0]>-0.04:    #if y_predict——Joule-Thomson close to zero
        ###print(y_predict[index][0], ' ',index)      #print y_predict
        # Reverse Normalization
        
        rho_li.append(x_test[index,0])              #for index print rho
        tem_li.append(x_test[index,1])              #for index print T
     

fig22 = plt.figure()
ax2 = fig22.add_subplot()
ax2.scatter(rho_li, tem_li)  #✔
plt.xlabel(r'$Density$')
plt.ylabel(r'$Temperature$')
plt.savefig("rho vs T at JTIC.png")
plt.close('all')

JTIC_rho = pd.DataFrame(rho_li)
JTIC_rho['tem_li'] = tem_li
JTIC_rho.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\JTIC_rho.xls')


'''
Block  Plot Error bar
'''
error = y_test - y_predict
plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.savefig("ErrorsHistograms_test.png")
plt.close('all')

pd.DataFrame(error).to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\error.xls')

'''
Block  JTIC FOR PRESSURE AND TEMPERATURE phase diagram
'''

tem_li = np.around(tem_li,3)

df2=pd.read_csv('C:\\Users\\LIHUI ZHAO\\rho-T-P.csv')

d2 = np.float32(df2.values)

d2, Pressure = np.split(d2,[2],1)

Density, Temperature = np.split(d2, [1], 1)

Density = np.around(Density,3)
Temperature = np.around(Temperature,3)


p_li=[]
T_li=[]
for index in range(len(Density)):
    for i in range(len(rho_li)):
        if Density[index]==rho_li[i] and Temperature[index]==tem_li[i]:
            T_li.append(tem_li[i])
            p_li.append(Pressure[index])


fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.scatter(p_li,T_li)  #✔
plt.xlabel(r'$Pressure$')
plt.ylabel(r'$Temperature$')
plt.savefig("P vs T at JTIC.png")
plt.close('all')

JTIC_p = pd.DataFrame(p_li)
JTIC_p['T_li'] = T_li
JTIC_p.to_excel('C:\\Users\\LIHUI ZHAO\\ANN_file\\Final_model_results\\JTIC_p.xls')


'''
Block  polyfit JTC points
'''
import matplotlib.pyplot as plt  
import numpy as np  
from scipy.interpolate import make_interp_spline  

p_li = pd.DataFrame(p_li)
p_li=p_li.iloc[:,0]
p_li=p_li.values
p_li.tolist()

z1 = np.polyfit(T_li, p_li, 2) 
param = np.poly1d(z1) 
print(param) 

z = np.polyval(param, np.linspace(min(T_li), max(T_li), 1000))
plt.scatter(T_li, p_li, c='r')
plt.plot(np.linspace(min(T_li), max(T_li),1000), z, linestyle='-')
plt.xlabel(r'$Temperature$')  
plt.ylabel(r'$Pressure$') 
plt.savefig("Joule-Thomson Inversion Curve.png")
plt.show()

