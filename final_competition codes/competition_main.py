from competition_utils import *

#initial variables
samples_path = 'DIS_lab_LoS/samples/' #path to the data
train_path = 'competition data/balanced data/train_data/'
test_path = 'competition data/balanced data/test_data/'
val_path = 'competition data/balanced data/validation_data/'
numb_users = 1                        #dont change
matplotlib.use('Agg')                 #images will not be plotted, intead they are gonna be saved in a folder
results_path = ''

#modes, dont change, since i have taken some functions from previous code this variables are needed for proper code execution
balanced_data = True                #selection by norm, user data will become less balanced
cnn = True
save_images = False
load = True
hist = None
path = '' 

#create list of users
users_list = [USER(i+1) for i in range(numb_users)]

#load data and assign model
print("Loading train data from ("+train_path+") folder...")
# tmp = os.listdir(samples_path)        #tmp contains all filenames
train_data_list = os.listdir(train_path)
test_data_list = os.listdir(test_path)
val_data_list = os.listdir(val_path)
#selection by norm of a sample
# if balanced_data == True and numb_users == 4:
if balanced_data == True and numb_users == 1:
    for user in users_list:
        for f in train_data_list:
            x = np.load(train_path+f)
            user.x.append((np.float32(x.real), np.float32(x.imag)))
        for f in test_data_list:
            x = np.load(test_path+f)
            user.test_data.append((np.float32(x.real), np.float32(x.imag)))
        for f in val_data_list:
            x = np.load(val_path+f)
            user.val_data.append((np.float32(x.real), np.float32(x.imag)))
                
#convert to numpy array and reshape, dimension two separates data into real and imaginary
for user in users_list:
    user.x = np.array(user.x).reshape(len(user.x), 2, 64, 100, 1)
    user.test_data = np.array(user.test_data).reshape(len(user.test_data), 2, 64, 100, 1)
    user.val_data = np.array(user.val_data).reshape(len(user.val_data), 2, 64, 100, 1)
    
#assign model to user(s), 3 different ways available
# 1. Default autoencoder
for user in users_list:
    tensorflow.random.set_seed(1)#48787
    user.assign_cnn() #REGULAR AUTOENCODER
    hist = user.model.fit(user.x, user.x, epochs=20, batch_size = 32, validation_data=(user.val_data, user.val_data))
    #==========================================================================
    #task 1
    # user.autoencoder1()
    # user.autoencoder2()
    # user.autoencoder3()
    
    # hist1 = user.model1.fit(user.x, user.x, epochs=12, batch_size = 64, validation_data = (user.val_data, user.val_data))
    # hist2 = user.model2.fit(user.x, user.x, epochs=10, batch_size = 64, validation_data = (user.val_data, user.val_data))
    # hist3 = user.model3.fit(user.x, user.x, epochs=12, batch_size = 64, validation_data = (user.val_data, user.val_data))
    
    # print(user.model1.evaluate(user.test_data, user.test_data))
    # print(user.model2.evaluate(user.test_data, user.test_data))
    # print(user.model3.evaluate(user.test_data, user.test_data))
    #==========================================================================
    #task 3
    #sort of continued learning, this is a very basic approach for task 3
    
    #regular autoencoder will encode CSI data
    # user.autoencoder1()
    # user.autoencoder2()
    # user.autoencoder3()
    
    # hist1 = user.model1.fit(user.x, user.x, epochs=12, batch_size = 64, validation_data = (user.val_data, user.val_data))
    # hist2 = user.model2.fit(user.x, user.x, epochs=10, batch_size = 64, validation_data = (user.val_data, user.val_data))
    # hist3 = user.model3.fit(user.x, user.x, epochs=12, batch_size = 64, validation_data = (user.val_data, user.val_data))
    
    #get encoded features
    # encoded_100 = Model(inputs = user.model1.input, outputs = user.model1.get_layer('h7').output).predict(user.x[0:1000])
    # encoded_50 = Model(inputs = user.model2.input, outputs = user.model2.get_layer('h7').output).predict(user.x[0:1000])
    # encoded_25 = Model(inputs = user.model3.input, outputs = user.model3.get_layer('h7').output).predict(user.x[0:1000])

    #as described in competition specs document
    # user.assign_flex_decoder()
    # choice_list = [0,0,1,1,1,2,2,2,2,2]
    # for i in range(20):
    #     print('iteration', i)
    #     c = np.random.choice(choice_list)
    #     if c == 1:
    #         user.flex_decoder.fit(encoded_50, user.x[0:1000], epochs = 1, batch_size = 32)
    #     elif c == 2:
    #         user.flex_decoder.fit(encoded_100, user.x[0:1000], epochs = 1, batch_size = 32)
    #     else:
    #         user.flex_decoder.fit(encoded_25, user.x[0:1000], epochs = 1, batch_size = 32)
    
    # encoded_100_TEST = Model(inputs = user.model1.input, outputs = user.model1.get_layer('h7').output).predict(user.x[1000:2000])
    # encoded_50_TEST = Model(inputs = user.model2.input, outputs = user.model2.get_layer('h7').output).predict(user.x[1000:2000])
    # encoded_25_TEST = Model(inputs = user.model3.input, outputs = user.model3.get_layer('h7').output).predict(user.x[1000:2000])

    # print(user.flex_decode.evaluater(encoded_100_TEST, user.x[1000:2000]))
    # print(user.flex_decode.evaluater(encoded_50_TEST, user.x[1000:2000]))
    # print(user.flex_decode.evaluater(encoded_25_TEST, user.x[1000:2000]))
    #==========================================================================

#use plot_fft to check reconstruction