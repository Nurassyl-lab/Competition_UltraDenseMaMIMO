import numpy as np
import keras
import os
from keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from numpy.fft import fft, ifft, fftfreq, fft2
from numpy.random import seed
import tensorflow
# from datetime import date
from datetime import datetime
import matplotlib
import scipy
from sklearn.metrics import mean_squared_error
from numpy.random import permutation
t_size = 35
l_size = 25
tick_size = 20
#helps to load the data in a right order
filename = 'channel_measurement_'
user_pos = np.load('DIS_lab_LoS/user_positions.npy')
ant_pos = np.load('DIS_lab_LoS/antenna_positions.npy')

# def snr_complex_1(a, axis=0, ddof=0):
#     mag = np.abs(a[0,:,:,:] + (a[1,:,:,:]*1j)).reshape(64,100)
#     m = mag.mean(axis)
#     sd = mag.std(axis=axis, ddof=ddof)
#     return np.where(sd == 0, 0, m/sd)
def scheduler(epoch, lr):
    if epoch < 6:
        return lr
    else:
        return lr * np.exp(-0.1)

callback = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)

def save_model(model, path, model_name):
    model.save(path+model_name)
    
def evaluate_model(user, model, path, name):
    ev = model.evaluate(user.test_data, user.test_data)
    with open(path+name+'.txt', 'a') as f:
        f.write(str(ev))

def save_history(hist, path, name):
    np.save(path+name+'train_loss'+'.npy', hist.history['loss'])
    np.save(path+name+'val_loss'+'.npy', hist.history['val_loss'])
    plt.figure(figsize = (12, 8))
    plt.plot(range(len(hist.history['loss'])), hist.history['loss'], label = 'train loss')
    plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], label = 'val loss')
    plt.xlabel('epochs', fontsize = l_size)
    plt.ylabel('mse loss', fontsize = l_size)
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    plt.legend()
    plt.savefig(path+name+'.png')
    plt.close()
        
    
class Random_Dropout(keras.layers.Layer):# custom layer
    def __init__(self, units=32, input_dim=32):
        super(Random_Dropout, self).__init__()
        self.dropout1 = layers.Dropout(0.0, trainable = True)
        self.dropout2 = layers.Dropout(0.5, trainable = True)
        self.dropout3 = layers.Dropout(0.75, trainable = True)
        self.choice_list = [0,0,1,1,1,2,2,2,2,2]
        self.tr = tensorflow.Variable(np.random.choice(self.choice_list))
        
    def call(self, inputs):
        self.tr = np.random.choice(self.choice_list)
        print('tr = ', self.tr)
        if self.tr == 0:
            # self.randarr = np.random.choice([0, 1], size=(2,2560), p=[0.0, 1.0])
            x = self.dropout1(inputs)
        elif self.tr == 1:
            # self.randarr = np.random.choice([0, 1], size=(2,2560), p=[0.5, 0.5])
            x = self.dropout2(inputs)
        elif self.tr == 2:
            # self.randarr = np.random.choice([0, 1], size=(2,2560), p=[0.75, 0.25])
            x = self.dropout3(inputs)
        else:
            # self.randarr = np.random.choice([0, 1], size=(2,2560), p=[0.0, 1.0])
            x = self.dropout1(inputs)
        # inputs = tensorflow.reshape(inputs, [len(inputs), 2, 2560])
        # inputs = inputs * self.randarr
        # return tensorflow.reshape(inputs, [len(inputs), 2, 4, 5, 128])
        return x

def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()

def snr_complex(a):#output snr in dB
    mag = np.abs(a[0,:,:,:] + (a[1,:,:,:]*1j)).reshape(64,100)
    # print(np.min(mag))
    return 20*np.log10(np.mean(mag) / np.std(mag))

def SNR(a):
    mag = np.abs(a)
    return 20*np.log10(np.mean(mag) / np.std(mag))

def fit_dirstribution(x, user_numb, save_images, path, dist = ['beta']):#input is users_list[0].x
    from fitter import Fitter, get_common_distributions, get_distributions
    import random
    from scipy.stats import beta
    
    f_real = Fitter(x[:, 0, :, :, 0],
               distributions=dist, timeout=2000)
    f_imag = Fitter(x[:, 1, :, :, 0],
               distributions=dist, timeout=2000)
    f_real.fit()
    f_imag.fit()

    f_real.summary()    
    f_imag.summary()
    
    #here I get distribution parameters that can be send to a server
    a_r, b_r, loc_r, scale_r = f_real.get_best(method = 'sumsquare_error')[next(iter(f_real.get_best(method = 'sumsquare_error')))]
    a_i, b_i, loc_i, scale_i = f_imag.get_best(method = 'sumsquare_error')[next(iter(f_imag.get_best(method = 'sumsquare_error')))]
    
    #assume now you have a server that received parameters

    #create array of model input shape and fill it with normally random numbers
    server_data_r = np.array([[[random.uniform(0, 1) for i in range(100)] for j in range(64)] for k in range(1000)])
    server_data_i = np.array([[[random.uniform(0, 1) for i in range(100)] for j in range(64)] for k in range(1000)])
    
    #generate new synthetic data using distribution parameters that were parsed to the server
    server_data_r = beta.cdf(beta.ppf(server_data_r, a_r, b_r, loc_r, scale_r), a_r, b_r, loc_r, scale_r)
    server_data_i = beta.cdf(beta.ppf(server_data_i, a_i, b_i, loc_i, scale_i), a_i, b_i, loc_i, scale_i)

    server_data = np.vstack((server_data_r, server_data_i)).reshape(1000, 2, 64, 100, 1)
    act_function = 'tanh'
    model = Sequential()
    model.add(keras.Input(shape=(2, 64, 100, 1)))
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'enc'))#32 50
    model.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h2'))
    model.add(layers.UpSampling3D((1, 2, 2), name = 'h3'))
    model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'dec'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(server_data, server_data, epochs = 10, batch_size = 64, validation_split=0.2)

    decoded_data = model.predict(x[-1000:])

    plot_fft(x[-1000:], user_numb, t = '', sample = 500, ant = 0, over_subc = True, subc = 0, over_ants = False, save = save_images, path = path) 
    plot_fft(decoded_data, user_numb, t = '', sample = 500, ant = 0, over_subc = True, subc = 0, over_ants = False, save = save_images, path = path) 

def intensity_plot(x, n, t = '', sample = 0, save = False, path = ''):
    # plot_fft(x, t = ' + FFT', sample = 0, ant = 0, over_subc = True)
    x = x.reshape(len(x), 2, 64, 100)
    data = x[sample, 0, :, :] + (x[sample, 1, :, :] * 1j)
    # print(data.shape)
    data = np.abs(data)
    # fig,= plt.subplots(figsize=(12, 6))
    plt.figure()
    plt.title("Intensity plot of sample "+str(sample), fontsize=20)
    # plt.imshow (data, interpolation='nearest', origin='lower')
    # plt.imshow(data, cmap = 'viridis')
    plt.imshow(data)
    plt.colorbar()
    if save:
        plt.savefig(path+'intensity_u'+str(n)+'.png')
        plt.close()

def bode_plot(x, n, t = '', sample = 0, ant = 0, over_subc = False, subc = 0, over_ants = False, save = False, path = ''): 
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
    plt.stem(20*np.log10(np.abs(Complex)), linefmt = 'grey', markerfmt='or')
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
    plt.stem(np.arctan(Complex.real / Complex.imag) * (180/np.pi), linefmt = 'grey', markerfmt='or')
    if over_subc == True:
        plt.ylabel('Phase of CSI signal in degrees', fontsize = 15)
        plt.xlabel('subcarriers', fontsize = 15)
    elif over_ants == True:
        plt.ylabel('Phase of CSI signal in degrees', fontsize = 15)
        plt.xlabel('antennas', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    if save:
        if over_subc:
            plt.savefig(path+'bodeplot_u'+str(n)+'_oversubc.png')
        else:
            plt.savefig(path+'bodeplot_u'+str(n)+'_overant.png')
        plt.close()
    
def plot_fft(x, n, t = '', sample = 0, ant = 0, over_subc = False, subc = 0, over_ants = False, save = False, path = ''):
    l = l_size - 7
    if over_subc == True:
        fig = plt.figure(figsize=(16,10))
        fig.suptitle('Plot of signal over 100 subcarriers of antenna '+str(ant) + " " + t, fontsize=20)
        Complex = x[sample, 0, ant, :] + (x[sample, 1, ant, :] * 1j)
        Complex = Complex.reshape(100)
    elif over_ants == True:
        fig.suptitle('Plot of signal over 64 antennas and subcarrier '+str(subc) + " " + t, fontsize=20)
        fig = plt.figure(figsize=(16,10))
        Complex = x[sample, 0, :, subc] + (x[sample, 1, :, subc] * 1j)
        Complex = Complex.reshape(x.shape[2])
    else:
        return
    complex_fft = fft(Complex)
    freq = fftfreq(len(Complex))
    plt.subplot(212)
    plt.stem(freq, np.abs(complex_fft), linefmt = 'grey', markerfmt="or")
    plt.xlabel('Freq (Hz)', fontsize = l)
    plt.ylabel('FFT Amplitude |X(freq)|', fontsize = l)
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    plt.subplot(221)
    plt.stem(range(len(complex_fft)), 20*np.log10(np.abs(ifft(complex_fft))), linefmt = 'grey', markerfmt='or')
    plt.stem(range(len(Complex)), 20*np.log10(np.abs(Complex)), linefmt = 'grey', markerfmt='or')
    plt.xlabel('subcarriers', fontsize = l)
    plt.ylabel('Magnitude(dB)', fontsize = l)
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    # plt.tight_layout()
    plt.subplot(222)
    plt.stem(range(len(complex_fft)), np.arctan(ifft(complex_fft).imag / ifft(complex_fft).real) * (180/np.pi), linefmt = 'grey', markerfmt='or')
    plt.stem(range(len(Complex)), np.arctan(Complex.imag / Complex.real)* (180/np.pi), linefmt = 'grey', markerfmt='or')
    plt.xlabel('subcarriers', fontsize = l)
    plt.ylabel('Phase(degrees)', fontsize = l)
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    # plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(path+'fft_u'+str(n)+'.png')
        plt.close()

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
        self.val_data = []
        self.model_enc = None
        self.model_dec = None
        self.model_im = None
        self.n = '000000'
        self.hist = []
        # self.t_hist = []
        # self.model_dec1 = None
        # self.model_dec2 = None
        # self.model_dec3 = None
        
        self.dec_im = []

    def plot_subc(self, ant, subc):
        plt.figure()
        plt.plot(range(len(self.test_data[:,ant,subc])), self.test_data[:,ant,subc])
        plt.ylabel("signal strength")
        plt.xlabel("samples")
        
    def assign_cnn(self, act_function = 'tanh', lr = 0.0001):  
        self.model = Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=(2, 64, 100, 1)))
        self.model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))#32 50
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3'))
        self.model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4'))#16 25
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5'))
        self.model.add(layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6'))#8 5
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7'))
        # self.model.add(layers.MaxPooling3D((1, 2, 1), padding='same', name = 'encoder'))#4 5
        
        # self.model.add(layers.Conv3D(256, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
        # self.model.add(layers.UpSampling3D((1, 2, 1), name = 'h9'))#8 5
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
        self.model.add(layers.UpSampling3D((1, 2, 5), name = 'h11'))#16, 25
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#32, 50
        self.model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
        self.model.add(layers.UpSampling3D((1, 2, 2), name = 'h15'))
        self.model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')
    
    #do not use this
    def task2(self, act_function = 'tanh', lr = 0.001):  
        input_x = keras.Input(shape=(2, 64, 100, 1))
        x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)#32 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)#16 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
        x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)#8 5
        encoded = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)
        # encoded = layers.MaxPooling3D((1, 2, 1), padding='same', name = 'encoded')(x)#4 5
        # encoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'encoded')(x)
        dropout = Random_Dropout()
        x = dropout(encoded)
        # x = layers.Conv3D(256, (1, 3, 3), activation=act_function, padding='same', name = 'h8')(x)
        # x = layers.UpSampling3D((1, 2, 1), name = 'h9')(x)#8 5
        x = layers.Conv3D(256, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
        x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)#16, 25
        x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)#32, 50
        x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
        decoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)
        self.task2_model = Model(input_x, decoded)
        self.task2_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')  
        
    # def assign_decoder1(self, act_function = 'tanh'):#100
    #     self.model_dec1 = Sequential()
    #     self.model_dec1.add(keras.layers.InputLayer(input_shape=(2, 4, 5, 1)))
    #     self.model_dec1.add(layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
    #     self.model_dec1.add(layers.UpSampling3D((1, 2, 1), name = 'h9'))#8 5
    #     self.model_dec1.add(layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
    #     self.model_dec1.add(layers.UpSampling3D((1,1, 2, 2), name = 'h13'))#32, 50
    #     self.model_dec1.add(layers.Conv3D(64, (1, 3 2, 5), name = 'h11'))#16, 25
    #     self.model_dec1.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
    #     self.model_dec1.add(layers.UpSampling3D((, 3), activation=act_function, padding = 'same',name = 'h14'))
    #     self.model_dec1.add(layers.UpSampling3D((1, 2, 2), name = 'h15'))
    #     self.model_dec1.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
    #     self.model_dec1.compile(optimizer='adam', loss = 'mse')
       
    def assign_flex_decoder(self, act_function = 'tanh', lr = 0.001):#50
        inputs = keras.Input(shape=(2, None, None, 64))
        l = inputs
        l = layers.Conv3D(128, (1, 3, 3), activation='tanh', padding='same')(l)
        l = layers.UpSampling3D((1, 2, 2))(l)
        l = layers.Conv3D(128, (1, 3, 3), activation='tanh', padding='same')(l)
        l = layers.GlobalMaxPooling3D()(l)
        l = layers.Dense(8*5*2, activation='tanh')(l)
        l = layers.Reshape((2, 8, 5, 1))(l)
        l = layers.UpSampling3D((1, 2, 5))(l)#16 25
        l = layers.Conv3D(128, (1, 3, 3), activation='tanh', padding='same')(l)
        l = layers.UpSampling3D((1, 2, 2))(l)#32 50
        l = layers.Conv3D(128, (1, 3, 3), activation='tanh', padding='same')(l)
        l = layers.UpSampling3D((1, 2, 2))(l)#64 100
        outputs = layers.Conv3D(1, (1, 3, 3), activation='tanh', padding='same')(l)
        self.flex_decoder = Model(inputs=inputs, outputs=outputs)
        self.flex_decoder.compile(loss='mse', optimizer='adam')
        self.flex_decoder.summary()
        
    def autoencoder1(self, act_function = 'tanh', lr = 0.001):
        input_x = keras.Input(shape=(2, 64, 100, 1))
        x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)#32 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)#16 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
        x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)#8 5
        encoded = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)

        x = layers.Dropout(0.0, trainable = True)(encoded)
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
        x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)#16, 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)#32, 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
        decoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)
        self.model1 = Model(input_x, decoded)
        self.model1.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse') 
        
    def autoencoder2(self, act_function = 'tanh', lr = 0.001):
        input_x = keras.Input(shape=(2, 64, 100, 1))
        x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)#32 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)#16 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
        x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)#8 5
        encoded = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)
        
        x = layers.Dropout(0.5, trainable = True)(encoded)
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
        x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)#16, 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)#32, 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
        decoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)
        self.model2 = Model(input_x, decoded)
        self.model2.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse') 
    
    def autoencoder3(self, act_function = 'tanh', lr = 0.001):
        input_x = keras.Input(shape=(2, 64, 100, 1))
        x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)#32 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
        x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)#16 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
        x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)#8 5
        encoded = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)
        
        x = layers.Dropout(0.75, trainable = True)(encoded)
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
        x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)#16, 25
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)#32, 50
        x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
        x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
        decoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)
        self.model3 = Model(input_x, decoded)
        self.model3.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse') 
        # input_x = keras.Input(shape=(2, 64, 100, 1))
        # x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_x)
        # x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)#32 50
        # x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
        # x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)#16 25
        # x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
        # x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)#8 5
        # x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)
        # x = layers.MaxPooling3D((1, 2, 1), padding='same', name = '7.1')(x)#4 5
        # x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7.2')(x)
        # encoded = layers.GlobalMaxPooling3D(name = 'encoded')(x)

        # x = layers.Dense(1 * 5 * 2)(encoded)
        # x = layers.Reshape((2, 1, 5, 1))(x)
        # x = layers.Conv3D(256, (1, 3, 3), activation=act_function, padding='same', name = 'h7.5')(x)
        # x = layers.UpSampling3D((1, 2, 1), name = 'h7.6')(x)#2 5
        # x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h7.3')(x)
        # x = layers.UpSampling3D((1, 2, 1), name = 'h7.4')(x)#4 5
        # x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h8')(x)
        # x = layers.UpSampling3D((1, 2, 1), name = 'h9')(x)#8 5
        # x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
        # x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)#16, 25
        # x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
        # x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)#32, 50
        # x = layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
        # x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
        # decoded = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)
        # self.model3 = Model(input_x, decoded)
        # self.model3.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse') 

    def assign_decoder1(self, act_function = 'tanh', lr = 0.001):#20
        self.model_dec1 = Sequential()
        self.model_dec1.add(keras.layers.InputLayer(input_shape=(2, 4, 5, 256)))
        self.model_dec1.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h6'))
        self.model_dec1.add(layers.UpSampling3D((1, 2, 1), name = 'h7'))#8 5
        self.model_dec1.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
        self.model_dec1.add(layers.UpSampling3D((1, 2, 5), name = 'h9'))#16 25
        self.model_dec1.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
        self.model_dec1.add(layers.UpSampling3D((1, 2, 2), name = 'h11'))#32, 50
        self.model_dec1.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
        self.model_dec1.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#64, 100
        self.model_dec1.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
        self.model_dec1.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model_dec1.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')

    def assign_decoder2(self, act_function = 'tanh', lr = 0.001):#10
        self.model_dec2 = Sequential()
        self.model_dec2.add(keras.layers.InputLayer(input_shape=(2, 2, 5, 256)))
        self.model_dec2.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h6'))
        self.model_dec2.add(layers.UpSampling3D((1, 4, 1), name = 'h7'))#8 5
        self.model_dec2.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
        self.model_dec2.add(layers.UpSampling3D((1, 2, 5), name = 'h9'))#16 25
        self.model_dec2.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
        self.model_dec2.add(layers.UpSampling3D((1, 2, 2), name = 'h11'))#32, 50
        self.model_dec2.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
        self.model_dec2.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#64, 100
        self.model_dec2.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
        self.model_dec2.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model_dec2.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')
        
    def assign_decoder3(self, act_function = 'tanh', lr = 0.001):#5
        self.model_dec3 = Sequential()
        self.model_dec3.add(keras.layers.InputLayer(input_shape=(2, 1, 5, 256)))
        self.model_dec3.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h6'))
        self.model_dec3.add(layers.UpSampling3D((1, 8, 1), name = 'h7'))#8 5
        self.model_dec3.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
        self.model_dec3.add(layers.UpSampling3D((1, 2, 5), name = 'h9'))#16 25
        self.model_dec3.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
        self.model_dec3.add(layers.UpSampling3D((1, 2, 2), name = 'h11'))#32, 50
        self.model_dec3.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
        self.model_dec3.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#64, 100
        self.model_dec3.add(layers.Conv3D(128, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
        self.model_dec3.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
        self.model_dec3.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')


def FFT_2D(x):
    x = x[0, :, :, :] + (x[1, :, :, :] * 1j)
    x = x.reshape(x.shape[0], 100)
    fft_2d = fft2(x)
    return fft_2d

# act_function = 'tanh'
# model = Sequential()
# model.add(keras.layers.InputLayer(input_shape=(2, 64, 100, 1)))
# model.add(layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
# model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))#32 50
# model.add(layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h3'))
# model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4'))#16 25
# model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h5'))
# model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h6'))#8 12
# model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'tmp1'))
# model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'tmp2'))#4 6
# model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'tmp3'))
# model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'tmp4'))#2 3
# ###############################################################################################
# model.add(layers.Conv3D(64, (1, 1, 1), activation=act_function, padding='same', name = 'h7'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'h8'))#4 6
# model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h9'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'h10'))#8, 12
# model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h11'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'h12'))#16 24
# model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'tmp5'))#32 48
# model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'tmp6'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'tmp7'))#64 48
# model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'tmp8'))
# model.compile(optimizer='adam', loss = 'mse')
# act_function = 'tanh'
# input_enc = keras.Input(shape=(2, 64, 100, 1))
# x = layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h1')(input_enc)
# x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2')(x)
# x = layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
# x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)
# x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h5')(x)
# x = layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6')(x)
# x = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'h7')(x)
# enc = (layers.MaxPooling3D((1, 2, 1), padding='same', name = 'encoded'))(x)

# # input_dec = keras.Input(shape=(2, 4, 5, 1))
# x = layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h8')(enc)
# x = layers.UpSampling3D((1, 2, 1), name = 'h9')(x)
# x = layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
# x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)
# x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
# x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)
# x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
# x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
# dec = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)

# auto = Model(input_enc, dec)
# auto.compile(optimizer='adam', loss='mse')

# input_dec = keras.Input(shape=(2, 4, 5, 1))
# x = layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h8')(input_dec)
# x = layers.UpSampling3D((1, 2, 1), name = 'h9')(x)
# x = layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
# x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)
# x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
# x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)
# x = layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14')(x)
# x = layers.UpSampling3D((1, 2, 2), name = 'h15')(x)
# dec = layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded')(x)

# decoder = Model(input_dec, dec)
# decoder.compile(optimizer='adam', loss='mse')
# Model(inputs=auto.get_layer("encoded").output, outputs=decoder)


# model = Sequential()
# model.add(keras.layers.InputLayer(input_shape=(2, 6400, 1)))
# model.add(layers.Conv2D(8, (1, 3), activation=act_function, padding='same', name = 'h1'))
# model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h2'))#3200
# model.add(layers.Conv2D(16, (1, 3), activation=act_function, padding='same', name = 'h3'))
# model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h4'))#1600
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='same', name = 'h5'))
# model.add(layers.MaxPooling2D((1, 4), padding='same', name = 'h6'))#400
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='same', name = 'h7'))
# model.add(layers.MaxPooling2D((1, 4), padding='same', name = 'h8'))#100
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='same', name = 'h9'))
# model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h10'))#50
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='valid', name = '1'))#48
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='valid', name = '2'))#46
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='valid', name = '3'))#44
# model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='valid', name = '4'))#42
# model.add(layers.Conv2D(1, (1, 3), activation=act_function, padding='valid', name = 'enc'))#40
# # model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h12'))#100
# # model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='same', name = 'h13'))
# # model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h14'))#20
# # model.add(layers.Conv2D(32, (1, 3), activation=act_function, padding='same', name = 'h15'))
# # model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'h16'))#10
# # model.add(layers.Conv2D(1, (1, 3), activation=act_function, padding='same', name = 'encoded'))
# ###############################################################################################
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h17'))#40
# model.add(layers.MaxPooling2D((1, 2), padding='same', name = 'lmao'))#20
# # model.add(layers.UpSampling2D((1, 2), name = 'h18'))#20
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h19'))
# model.add(layers.UpSampling2D((1, 5), name = 'h20'))#100
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h21'))
# model.add(layers.UpSampling2D((1, 2), name = 'h22'))#200
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h23'))
# model.add(layers.UpSampling2D((1, 2), name = 'h24'))#400
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h25'))
# model.add(layers.UpSampling2D((1, 2), name = 'h26'))#800
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h27'))
# model.add(layers.UpSampling2D((1, 2), name = 'h28'))#1600
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h29'))
# model.add(layers.UpSampling2D((1, 2), name = 'h30'))#3200
# model.add(layers.Conv2D(64, (1, 3), activation=act_function, padding='same', name = 'h31'))
# model.add(layers.UpSampling2D((1, 2), name = 'h32'))#6400
# model.add(layers.Conv2D(1, (1, 3), activation=act_function, padding='same', name = 'decoder'))

# model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse')

# model = Sequential()
# model.add(keras.layers.InputLayer(input_shape=(2, 4, 5, 1)))
# model.add(layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h8'))
# model.add(layers.UpSampling3D((1, 2, 1), name = 'h9'))#8 5

# model.add(layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
# model.add(layers.UpSampling3D((1, 2, 5), name = 'h11'))#16, 25

# model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
# model.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#32, 50
# model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
# model.add(layers.UpSampling3D((1, 1, 2), name = 'h15'))
# model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'out'))
# model.compile(optimizer='adam', loss = 'mse')

# hist = model.fit(users_list[0].x.reshape(10000, 2, 6400, 1), users_list[0].x.reshape(10000, 2, 6400, 1), epochs = 10, batch_size = 64, validation_split = 0.2)
# evs = model.evaluate(users_list[0].x.reshape(10000, 2, 6400, 1), users_list[0].x.reshape(10000, 2, 6400, 1))
# norma = np.linalg.norm(users_list[0].x.reshape(10000, 2, 6400, 1))
# nmse = evs / norma
# dec = model.predict(users_list[0].x.reshape(10000, 2, 6400, 1))
# plot_fft(users_list[0].x, user.number, t = '', sample = 500, ant = 0, over_subc = True, subc = 0, over_ants = False, save = save_images, path = path) 



# input_enc = keras.Input(shape=(28, 28, 1))
# x = layers.Conv2D(8, (2, 2), activation=act_function, padding='same', name = 'h1')(input_enc)
# x = layers.MaxPooling2D((2, 1), padding='same', name = 'h2')(x)
# # x = layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h3')(x)
# # x = layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4')(x)


# # input_dec = keras.Input(shape=(2, 4, 5, 1))
# # x = layers.Conv3D(8, (1, 3, 3), activation=act_function, padding='same', name = 'h8')(enc)
# # x = layers.UpSampling3D((1, 2, 1), name = 'h9')(x)
# # x = layers.Conv3D(16, (1, 3, 3), activation=act_function, padding='same', name = 'h10')(x)
# # x = layers.UpSampling3D((1, 2, 5), name = 'h11')(x)
# # x = layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h12')(x)
# # x = layers.UpSampling3D((1, 2, 2), name = 'h13')(x)
# x = layers.Conv2D(16, (3, 1), activation=act_function, padding = 'same',name = 'h14')(x)
# x = layers.UpSampling2D((2, 1), name = 'h15')(x)
# dec = layers.Conv2D(1, (3, 1), activation=act_function, padding='same', name = 'decoded')(x)

# input_x = layers.Input(shape=(28, 28, 1))

# # Encoder
# x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_x)
# x = layers.MaxPooling2D((2, 2), padding="same")(x)
# x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
# x = layers.MaxPooling2D((2, 2), padding="same", name = 'enc')(x)
# x = layers.Conv2D(1, (3, 3), padding="same", name = 'enc1')(x)
# # Decoder
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# # Autoencoder
# auto = Model(input_x, x)
# auto.compile(optimizer="adam", loss="binary_crossentropy")
# auto.summary()


# input_dec = layers.Input(shape=(7, 7, 1))
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(input_dec)
# x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
# x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# # Autoencoder
# decoder = Model(input_dec, x)
# decoder.compile(optimizer="adam", loss="binary_crossentropy")
# decoder.summary()

# from keras.datasets import fashion_mnist
# (x_train, _), (_,_) = fashion_mnist.load_data()
# x = x_train.reshape(60000, 28, 28, 1)[0:10000] / 255.0

# auto.fit(x, x, epochs = 2)
# decs = Model(inputs = auto.input, outputs = auto.get_layer('enc1').output).predict(x)

# decoder.fit(decs, x, epochs = 2)





# tf.random.set_seed(0)Adam




