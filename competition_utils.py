import keras
from keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from numpy.fft import fft, ifft, fftfreq
from numpy.random import seed
import tensorflow
# from datetime import date
from datetime import datetime


#helps to load the data in a right order
filename = 'channel_measurement_'
user_pos = np.load('DIS_lab_LoS/user_positions.npy')
ant_pos = np.load('DIS_lab_LoS/antenna_positions.npy')

def intensity_plot(x, t = '', sample = 0):
    # plot_fft(x, t = ' + FFT', sample = 0, ant = 0, over_subc = True)
    x = x.reshape(len(x), 2, 64, 100)
    data = x[sample, 0, :, :] + (x[sample, 1, :, :] * 1j)
    # print(data.shape)
    data = np.abs(data)
    # fig,= plt.subplots(figsize=(12, 6))
    plt.figure()
    plt.title("Intensity plot of sample "+str(sample), fontsize=20)
    # plt.imshow (data, interpolation='nearest', origin='lower')
    plt.imshow(data, cmap = 'viridis')
    plt.colorbar()

def bode_plot(x, t = '', sample = 0, ant = 0, over_subc = False, subc = 0, over_ants = False): 
    if over_subc == True:
        fig = plt.figure(figsize=(12,6))
        fig.suptitle('Plot of signal over 100 subcarriers of antenna '+str(ant) + " " + t, fontsize=20)
        Complex = x[sample, 0, ant, :] + (x[sample, 1, ant, :] * 1j)
        Complex = Complex.reshape(100)
    elif over_ants == True:
        fig.suptitle('Plot of signal over 64 antennas and subcarrier '+str(subc) + " " + t, fontsize=20)
        fig = plt.figure(figsize=(12,6))
        Complex = x[sample, 0, :, subc] + (x[sample, 1, :, subc] * 1j)
        Complex = Complex.reshape(64)
    else:
        return
    
    plt.subplot(121)
    plt.title('Magnitude', fontsize = 15)
    plt.grid(True)
    plt.stem(np.abs(Complex), linefmt = 'grey', markerfmt='xr')
    if over_subc == True:
        plt.ylabel('Magnitude of CSI signal in dB', fontsize = 15)
        plt.xlabel('subcarriers', fontsize = 15)
    elif over_ants == True:
        plt.ylabel('Magnitude of CSI signal in dB', fontsize = 15)
        plt.xlabel('antennas', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    plt.subplot(122)
    plt.title('Phase', fontsize = 15)
    plt.grid(True)
    plt.stem(np.arctan(Complex.real / Complex.imag) * (180/np.pi), linefmt = 'grey', markerfmt='xr')
    if over_subc == True:
        plt.ylabel('Phase of CSI signal in degrees', fontsize = 15)
        plt.xlabel('subcarriers', fontsize = 15)
    elif over_ants == True:
        plt.ylabel('Phase of CSI signal in degrees', fontsize = 15)
        plt.xlabel('antennas', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
def plot_fft(x, t = '', sample = 0, ant = 0, over_subc = False, subc = 0, over_ants = False):
    if over_subc == True:
        fig = plt.figure(figsize=(12,6))
        fig.suptitle('Plot of signal over 100 subcarriers of antenna '+str(ant) + " " + t, fontsize=20)
        Complex = x[sample, 0, ant, :] + (x[sample, 1, ant, :] * 1j)
        Complex = Complex.reshape(100)
    elif over_ants == True:
        fig.suptitle('Plot of signal over 64 antennas and subcarrier '+str(subc) + " " + t, fontsize=20)
        fig = plt.figure(figsize=(12,6))
        Complex = x[sample, 0, :, subc] + (x[sample, 1, :, subc] * 1j)
        Complex = Complex.reshape(64)
    else:
        return
    
    complex_fft = fft(Complex)
    freq = fftfreq(len(Complex))
    plt.subplot(212)
    plt.stem(freq, np.abs(complex_fft), linefmt = 'grey', markerfmt="xr")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.subplot(221)
    plt.stem(range(len(complex_fft)), 20*np.log10(np.abs(ifft(complex_fft))), linefmt = 'grey', markerfmt='xr')
    plt.stem(range(len(Complex)), 20*np.log10(np.abs(Complex)), linefmt = 'grey', markerfmt='xr')
    plt.xlabel('subcarriers')
    plt.ylabel('Magnitude(dB)')
    # plt.tight_layout()
    plt.subplot(222)
    plt.stem(range(len(complex_fft)), np.arctan(ifft(complex_fft).imag / ifft(complex_fft).real) * (180/np.pi), linefmt = 'grey', markerfmt='xr')
    plt.stem(range(len(Complex)), np.arctan(Complex.imag / Complex.real)* (180/np.pi), linefmt = 'grey', markerfmt='xr')
    plt.xlabel('subcarriers')
    plt.ylabel('Phase(degrees)')
    # plt.tight_layout()
    plt.show()

def plot_two_CSI_curves(x, y, ant = 0, t = "", labelx = "", labely = ""):
    plt.figure(figsize=(10, 8))
    plt.title(t)
    plt.grid(True)
    plt.plot(range(len(x[:,0])), x[:,0], label = labelx)
    plt.plot(range(len(y[:,0])), y[:,0], label = labely)
    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.07), ncol = 2)
    # plt.legend()
    plt.xlabel('samples')
    plt.ylabel('pilot signal value')
    plt.show()

class USER:
    def __init__(self, n):
        self.number = n
        self.x = []
        self.train_data = []
        self.test_data = []
        self.model = None
        self.model_im = None
        self.n = '000000'
        # self.t_hist = []
        self.dec = []
        self.dec_im = []

    def plot_subc(self, ant, subc):
        plt.figure()
        plt.plot(range(len(self.test_data[:,ant,subc])), self.test_data[:,ant,subc])
        plt.ylabel("signal strength")
        plt.xlabel("samples")

    def split_data(self, subc):
        self.x = self.x.reshape(len(self.x), self.x.shape[1] * self.x.shape[2])
        
    
    def train(self, ep, bs, sh, enc_dim):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(self.x.shape[1], )))
        self.model.add(layers.Dense(enc_dim, activation = 'tanh', name = 'h1'))
        self.model.add(layers.Dense(self.x.shape[1], activation = 'tanh', name = 'out'))
        self.model.compile(optimizer = 'adam', loss = 'mse')
        
        # self.t_hist = self.model.fit(self.x[0:-1000], self.x[0:-1000], validation_split = 0.2, epochs = ep, batch_size = bs, shuffle=sh)
        # self.t_hist = self.model.fit(self.x[0:-1000], self.x[0:-1000], epochs = ep, batch_size = bs, shuffle=sh)
        
    def train_im(self, ep, bs, sh, enc_dim):
        self.model_im = Sequential()
        self.model_im.add(keras.Input(shape=(self.x.shape[1], )))
        self.model_im.add(layers.Dense(enc_dim, activation = 'tanh', name = 'h1'))
        self.model_im.add(layers.Dense(self.x.shape[1], activation = 'tanh', name = 'out'))
        self.model_im.compile(optimizer = 'adam', loss = 'mse')
        
    def assign_cnn(self, act_function = 'tanh'):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(2, 64, 100, 1)))
        self.model.add(layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))#32 50
        self.model.add(layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h3'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4'))#16 25
        
        self.model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h5'))
        self.model.add(layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6'))#8 5
        
        # self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'enc1'))
        # self.model.add(layers.MaxPooling3D((1, 8, 5), padding='same', name = 'enc2'))#enc 1, 1
        
        # self.model.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'enc3'))
        # self.model.add(layers.UpSampling3D((1, 8, 5), name = 'enc4'))#8, 5
        
        self.model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h7'))
        self.model.add(layers.UpSampling3D((1, 2, 5), name = 'h8'))#16, 25
        
        self.model.add(layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h9'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h10'))#32, 50
        self.model.add(layers.Conv3D(8, (1, 3, 3), activation=act_function, padding = 'same',name = 'h11'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h12'))
        self.model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model.compile(optimizer='adam', loss='mse')
        
        
        
        # self.model = Sequential()
        # self.model.add(keras.Input(shape=(2, 64, 100, 1)))
        # self.model.add(layers.Conv3D(16, (1, 3, 3), activation='tanh', padding='same', name = 'h1'))
        # self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))
        # self.model.add(layers.Conv3D(32, (1, 3, 3), activation='tanh', padding='same', name = 'h3'))
        # self.model.add(layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h4'))
        # self.model.add(layers.Conv3D(32, (1, 3, 3), activation='tanh', padding='same', name = 'h5'))
        # self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'encoded'))
        # self.model.add(layers.Conv3D(64, (1, 3, 3), activation='tanh', padding='same', name = 'h6'))
        # self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h7'))
        # self.model.add(layers.Conv3D(64, (1, 3, 3), activation='tanh', padding='same', name = 'h8'))
        # self.model.add(layers.UpSampling3D((1, 2, 5), name = 'h9'))
        # self.model.add(layers.Conv3D(64, (1, 3, 3), activation='tanh', padding = 'same',name = 'h10'))
        # self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h11'))
        # self.model.add(layers.Conv3D(1, (1, 3, 3), activation='sigmoid', padding='same', name = 'decoded'))
        # self.model.compile(optimizer='adam', loss='mse')

        # users_list[0].train_cnn(10, 10, True)
        # users_list[0].model.summary()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        