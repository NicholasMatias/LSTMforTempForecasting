import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import*
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True
)
csv_path, _= os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
#print(df)

#::6 means to take in data every 6th, in this case every hour. Starts at the 6th row (index 5). 
df = df[5::6] 
#print(df)


#changes the numerical indexes to be time, format explains the ordering of the elements in the date. 
#format example: 31.12.2016 19:10:00
df.index= pd.to_datetime(df['Date Time'],format='%d.%m.%Y %H:%M:%S')

#print(df[:26])#this will get the first 26 hours. Can see that indexes are now the dates

#X[[[1],[2],[3],[4],[5]]] ->Y[6] numbers refer to hours. 
#X[[[2],[3],[4],[5],[6]]] ->Y[7]
#X[[[3],[4],[5],[6],[7]]] ->Y[8]


temp = df['T (degC)']

def df_to_X_Y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    Y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a  in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        Y.append(label)
    return np.array(X),np.array(Y)

X, Y = df_to_X_Y(temp,5)
#print(X.shape)
#print(Y.shape)
#print(X)
#print(Y)

X_train, Y_train = X[:60000], Y[:60000]
X_val, Y_val= X[60000:65000], Y[60000:65000]
X_test, Y_test = X[65000:],Y[65000:]
#print(f'{X_train.shape}\n{Y_train.shape}\n{X_val.shape}\n{Y_val.shape}\n{X_test.shape}\n{Y_test.shape}')


model1 = Sequential()
model1.add(InputLayer((5,1)))
model1.add(LSTM(64))
model1.add(Dense(8,'relu'))
model1.add(Dense(1,'linear'))

#print(model1.summary())

cp = ModelCheckpoint('model1/', save_best_only=True)

model1.compile(loss=MeanSquaredError(), optimizer = Adam(learning_rate=0.0001),metrics=[RootMeanSquaredError()])


model1.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs =10,callbacks=[cp])

from tensorflow.keras.models import load_model

model1 = load_model('model1/')



train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': Y_train})
print(train_results)

import matplotlib.pyplot as plt 

figure, axis = plt.subplots(1,3)


#see how it did on test data
axis[0].plot(train_results['Train Predictions'][:250])
axis[0].plot(train_results['Actuals'][:250])
axis[0].set_title("Train Predictions vs. Actuals")

#now lets check how it does on validation data

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions,'Actuals':Y_val})
#print(val_results)

axis[1].plot(val_results['Val Predictions'][:250])
axis[1].plot(val_results['Actuals'][:250])
axis[1].set_title("Val Predictions vs. Actuals")
#plt.show()


test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions,'Actuals':Y_test})
print(test_results)

axis[2].plot(test_results['Test Predictions'][:250])
axis[2].plot(test_results['Actuals'][:250])
axis[2].set_title("Test Predictions vs. Actuals")

plt.show()