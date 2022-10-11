from competition_utils import *

#initial variables
samples_path = 'DIS_lab_LoS/samples/' #path to the data

train1_path = 'competition data/unbalanced data/40_45/het_01/train_data/'
test1_path = 'competition data/unbalanced data/40_45/het_01/test_data/'

train2_path = 'competition data/unbalanced data/50_55/het_01/train_data/'
test2_path = 'competition data/unbalanced data/50_55/het_01/test_data/'

train3_path = 'competition data/unbalanced data/60_65/het_01/train_data/'
test3_path = 'competition data/unbalanced data/60_65/het_01/test_data/'


numb_users = 3                        #corresponds to number of models, either 1 or 4
size = 30000                          #size of user dataset
matplotlib.use('Agg')                 #images will not be plotted, intead they are gonna be saved in a folder
results_path = ''

#modes
balanced_data = False                #selection by norm, user data will become less balanced
cnn = True
save_images = False
load = True
hist = None

#create list of users
users_list = [USER(i+1) for i in range(numb_users)]

#load data and assign model
# print("Loading train data from ("+train_path+") folder...")
# tmp = os.listdir(samples_path)        #tmp contains all filenames
train1_data_list = os.listdir(train1_path)
test1_data_list = os.listdir(test1_path)

train2_data_list = os.listdir(train2_path)
test2_data_list = os.listdir(test2_path)

train3_data_list = os.listdir(train3_path)
test3_data_list = os.listdir(test3_path)
#selection by norm of a sample
for user in users_list:
    if user.number == 1:
        for f in train1_data_list:
            x = np.load(train1_path+f)
            user.x.append((np.float32(x.real), np.float32(x.imag)))
        for f in test1_data_list:
            x = np.load(test1_path+f)
            user.test_data.append((np.float32(x.real), np.float32(x.imag)))
            
    if user.number == 2:
        for f in train2_data_list:
            x = np.load(train2_path+f)
            user.x.append((np.float32(x.real), np.float32(x.imag)))
        for f in test2_data_list:
            x = np.load(test2_path+f)
            user.test_data.append((np.float32(x.real), np.float32(x.imag)))
            
    if user.number == 3:
        for f in train3_data_list:
            x = np.load(train3_path+f)
            user.x.append((np.float32(x.real), np.float32(x.imag)))
        for f in test3_data_list:
            x = np.load(test3_path+f)
            user.test_data.append((np.float32(x.real), np.float32(x.imag)))
                
#convert to numpy array and reshape, dimension two separates data into real and imaginary
for user in users_list:
    user.x = np.array(user.x).reshape(len(user.x), 2, 64, 100, 1)
    user.test_data = np.array(user.test_data).reshape(len(user.test_data), 2, 64, 100, 1)
    
#assign model to user(s), 4 different ways available
# 1. Default autoencoder
for user in users_list:
    tensorflow.random.set_seed(1)#48787
    user.assign_cnn()
    
hist1 = users_list[0].model.fit(users_list[0].x, users_list[0].x, epochs=10, batch_size = 64)
hist2 = users_list[1].model.fit(users_list[1].x, users_list[1].x, epochs=10, batch_size = 64)
hist3 = users_list[2].model.fit(users_list[2].x, users_list[2].x, epochs=10, batch_size = 64)    


# evals
print('model 1 evaluated on train_data of model 1')
users_list[0].model.evaluate(users_list[0].test_data, users_list[0].test_data)
print('model 1 evaluated on train_data of model 2')
users_list[0].model.evaluate(users_list[1].test_data, users_list[1].test_data)
print('model 1 evaluated on train_data of model 3')
users_list[0].model.evaluate(users_list[2].test_data, users_list[2].test_data)

print('model 2 evaluated on train_data of model 2')
users_list[1].model.evaluate(users_list[1].test_data, users_list[1].test_data)
print('model 2 evaluated on train_data of model 1')
users_list[1].model.evaluate(users_list[0].test_data, users_list[0].test_data)
print('model 2 evaluated on train_data of model 3')
users_list[1].model.evaluate(users_list[2].test_data, users_list[2].test_data)

print('model 3 evaluated on train_data of model 3')
users_list[2].model.evaluate(users_list[2].test_data, users_list[2].test_data)
print('model 3 evaluated on train_data of model 1')
users_list[2].model.evaluate(users_list[0].test_data, users_list[0].test_data)
print('model 1 evaluated on train_data of model 2')
users_list[2].model.evaluate(users_list[1].test_data, users_list[1].test_data)


#save models
path = 'competition data/different scenarios/heterogeneous/het_01/'
users_list[0].model.save(path+'model1')
users_list[1].model.save(path+'model2')
users_list[2].model.save(path+'model3')

#history
np.save(path+'history/model1_train_loss.npy', hist1.history['loss'])
np.save(path+'history/model2_train_loss.npy', hist3.history['loss'])
np.save(path+'history/model3_train_loss.npy', hist2.history['loss'])

#reconstruction
# plot_fft(users_list[0].x[0:10], user.number, t = '\n Original', sample = 0, ant = 0, over_subc = True, subc = 0, over_ants = False, save = save_images, path = path)





