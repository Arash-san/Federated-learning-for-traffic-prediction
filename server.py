import socket
from _thread import *
from numpy.core.numeric import tensordot
import pandas as pd
import numpy as np
import pickle
from scipy.stats.stats import Ttest_indResult
import seaborn as sns
import matplotlib.pyplot as plt

ServerSideSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
host = '0.0.0.0'
port = 7686
ThreadCount = 0
color = "#A0E7E5"
trained_data =  []
try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Server has been started. Clients can be connceted at anytime')
ServerSideSocket.listen(5)

def Sub_Plots2(df_1, df_2,title,m):
    fig, axes = plt.subplots(1, 2, figsize=(18,4), sharey=True,facecolor="#E57F84")
    fig.suptitle(title)
    
    pl_1=sns.lineplot(ax=axes[0],data=df_1,color=color)
    axes[0].set(ylabel ="Prediction")
    
    pl_2=sns.lineplot(ax=axes[1],data=df_2["Vehicles"],color="#2F5061")
    axes[1].set(ylabel ="Orignal")

def multi_threaded_client(connection,ip,port):
    
    while True:
        try:
            data = []
            while True:
                packet = connection.recv(4096)
                if not packet: break
                data.append(packet)
            
            data_variable = pickle.loads(b"".join(data))

            trained_data.append(data_variable)

            print("Recevied a trained model from "+ip+":"+str(port))

            trained = data_variable[0]
            test = data_variable[1]

            fig, axes = plt.subplots(1, 2, figsize=(18,4), sharey=True,facecolor="#E57F84")
            fig.suptitle("Pridictions And Orignals from "+ip+":"+str(port))
            
            pl_1=sns.lineplot(ax=axes[0],data=trained,color=color)
            axes[0].set(ylabel ="Prediction")
            
            pl_2=sns.lineplot(ax=axes[1],data=test["Vehicles"],color="#2F5061")
            axes[1].set(ylabel ="Orignal")
            
            # plt.figure(figsize=(12,5),facecolor="#627D78")
            # plt.plot(test, color=color,label="True Value",alpha=0.5 )
            # plt.title("GRU Traffic Prediction Vs True values")
            # plt.xlabel("DateTime")
            # plt.ylabel("Number of Vehicles")
            # plt.legend()

            plt.show()

        except:
            pass
             

while True:
    Client, address = ServerSideSocket.accept()
    print('A new node has been connected to the server: ' + address[0] + ':' + str(address[1]))
    print('Server will be ready to receive a trained model from this client')
    start_new_thread(multi_threaded_client, (Client,address[0],address[1], ))
    ThreadCount += 1

# ServerSideSocket.close()