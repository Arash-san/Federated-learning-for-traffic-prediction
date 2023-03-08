import socket
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD
from keras import callbacks


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("traffic.csv")
data.head()

try:
    ClientMultiSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    # host = input("Enter the IP Address of the server: ")
    host = "127.0.0.1"
    port = 7686

    print('Connecting to the server')
    
    ClientMultiSocket.connect((host, port))

except socket.error as e:
    print(str(e))

data["DateTime"]= pd.to_datetime(data["DateTime"])
data = data.drop(["ID"], axis=1)

df=data.copy() 

dataFrame_J = data.pivot(columns="Junction", index="DateTime")

dataFrame_1 = dataFrame_J[[('Vehicles', 2)]]

dataFrame_1.columns= dataFrame_1.columns.droplevel(level=1) 

def Normalize(df,col):
    average = df[col].mean()
    stdev = df[col].std()
    dataFrame_normalized = (df[col] - average) / stdev
    dataFrame_normalized = dataFrame_normalized.to_frame()
    return dataFrame_normalized, average, stdev

def Difference(df,col, interval):
    diff = []
    for i in range(interval, len(df)):
        value = df[col][i] - df[col][i - interval]
        diff.append(value)
    return diff

dataFrame_N1, av_J1, std_J1 = Normalize(dataFrame_1, "Vehicles")
Diff_1 = Difference(dataFrame_N1, col="Vehicles", interval=(24))
dataFrame_N1 = dataFrame_N1[24:]
dataFrame_N1.columns = ["Norm"]
dataFrame_N1["Diff"]= Diff_1
    
def Stationary_check(df):
    check = adfuller(df.dropna())
    print(f"ADF Statistic: {check[0]}")
    print("Critical Values:")
    for key, value in check[4].items():
        print('\t%s: %.3f' % (key, value))
    if check[0] > check[4]["1%"]:
        print("Time Series is Non-Stationary")
    else:
        print("Time Series is Stationary") 

Stationary_check(dataFrame_N1["Diff"])

dataFrame_J1 = dataFrame_N1["Diff"].dropna()
dataFrame_J1 = dataFrame_J1.to_frame()

def Split_data(df):
    training_size = int(len(df)*0.9)
    data_len = len(df)
    train, test = df[0:training_size],df[training_size:data_len] 
    train, test = train.values.reshape(-1, 1), test.values.reshape(-1, 1)
    return train, test


J1_train, J1_test = Split_data(dataFrame_J1)

def TnF(df):
    end_len = len(df)
    X = []
    y = []
    steps = 32
    for i in range(steps, end_len):
        X.append(df[i - steps:i, 0])
        y.append(df[i, 0])
    X, y = np.array(X), np.array(y)
    return X ,y

def FeatureFixShape(train, test):
    train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    test = np.reshape(test, (test.shape[0],test.shape[1],1))
    return train, test

X_trainJ1, y_trainJ1 = TnF(J1_train)
X_testJ1, y_testJ1 = TnF(J1_test)
X_trainJ1, X_testJ1 = FeatureFixShape(X_trainJ1, X_testJ1)

def GRU_model(X_Train, y_Train, X_Test):
    early_stopping = callbacks.EarlyStopping(min_delta=0.001,patience=10, restore_best_weights=True) 
    #callback delta 0.01 may interrupt the learning, could eliminate this step
    
    #The GRU model 
    model = Sequential()
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    #Compiling the model
    model.compile(optimizer=SGD(decay=1e-7, momentum=0.9),loss='mean_squared_error')
    model.fit(X_Train,y_Train, epochs=3, batch_size=150,callbacks=[early_stopping])
    pred_GRU= model.predict(X_Test)
    return pred_GRU


PredJ1 = GRU_model(X_trainJ1,y_trainJ1,X_testJ1)

def inverse_difference(last_ob, value):
    inversed = value + last_ob
    return inversed

recover1 = dataFrame_N1.Norm[-(len(PredJ1)+1):-1].to_frame()
recover1["Pred"]= PredJ1
Transform_reverssed_J1 = inverse_difference(recover1.Norm, recover1.Pred).to_frame()
Transform_reverssed_J1.columns = ["Pred_Normed"]
Final_J1_Pred = (Transform_reverssed_J1.values* std_J1) + av_J1
Transform_reverssed_J1["Pred_Final"] =Final_J1_Pred

result = [Transform_reverssed_J1["Pred_Final"],dataFrame_1[-(len(PredJ1)+1):-1]]

try:
    data_string = pickle.dumps(result)
    ClientMultiSocket.send(data_string)

except socket.error as e:
    print(str(e))



