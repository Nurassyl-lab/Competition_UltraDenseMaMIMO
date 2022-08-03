import keras
from keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

#helps to load the data in a right order
filename = 'channel_measurement_'

def pre_process(X):
    X = X.flatten()
    return X

def re_process(X):
    X = X.reshape(64,100)
    return X

def plot_two_CSI_curves(x, y, ant = 0, t = "", labelx = "", labely = ""):
    plt.figure(figsize=(10, 8))
    plt.title(t)
    plt.grid(True)
    plt.plot(range(len(x[:,0])), x[:,0], label = labelx)
    plt.plot(range(len(y[:,0])), y[:,0], label = labely)
    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.07), ncol = 2)
    # plt.legend()
    plt.xlabel('subcarries')
    plt.ylabel('signal strength')
    plt.show()

class USER:
    def __init__(self, n):
        self.number = n
        self.x = []
        self.train_data = []
        self.test_data = []
        self.model = None
        self.n = '000000'
        self.t_hist = []
        self.dec = []
    
    def load_data(self, path, size):
        tmp = int(self.n)
        for i in range(tmp, size*self.number):
            self.x.append(np.load(path+filename+self.n+'.npy'))
            self.n = self.n[0:len(self.n) - len(str(i))] + str(i)
        self.x = np.array(self.x)

    # def load_data(self, path, size):
    #     tmp = int(self.n)
    #     for i in range(tmp, size*self.number):
    #         if i < tmp + (size/5):
    #             self.test_data.append(np.load(path+filename+self.n+'.npy'))
    #         else:
    #             self.train_data.append(np.load(path+filename+self.n+'.npy'))
    #         self.n = self.n[0:len(self.n) - len(str(i))] + str(i)
    #     self.train_data = np.array(self.train_data)
    #     self.test_data = np.array(self.test_data)

    def plot_subc(self, ant, subc):
        plt.figure()
        plt.plot(range(len(self.test_data[:,ant,subc])), self.test_data[:,ant,subc])
        plt.ylabel("signal strength")
        plt.xlabel("samples")
    
    def split_data(self, subc):
        self.train_data = self.x[int(len(self.x) / 5):len(self.x),:,subc]
        self.test_data = self.x[0:int(len(self.x) / 5),:,subc]
    
    def train(self, ep, bs, sh, enc_dim):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(64, )))
        self.model.add(layers.Dense(10, activation = 'tanh', name = 'h1'))
        self.model.add(layers.Dense(64, activation = 'tanh', name = 'out'))
        self.model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
        
        self.t_hist = self.model.fit(self.train_data, self.train_data, validation_split = 0.2, epochs = ep, batch_size = bs, shuffle=sh)
        
        
        
        
        
        
        
        
        
        
        
        
        
        