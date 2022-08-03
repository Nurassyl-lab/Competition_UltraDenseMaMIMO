"Ultra Dense Indoor MaMIMO CSI Dataset"
from competition_utils import *

#inital variables
samples_path = 'DIS_lab_LoS/samples/'
numb_users = 1
size = int(252004 / numb_users) #size of user dataset

#load antenna position and user position for one user
ant_pos = np.load('DIS_lab_LoS/antenna_positions.npy')
if numb_users == 1:
    user_pos = np.load('DIS_lab_LoS/user_positions.npy')

#create list of users
users_list = [USER(i+1) for i in range(numb_users)]

#load data
#since there are memory problems, I load only 63000 samples
size = 63000
subcarry = 10#any number from [0, 99]
for user in users_list: user.load_data(samples_path, size)

#split data into train and test
for user in users_list: user.split_data(subc = subcarry)

#train users
#the model will be assigned and trained
ep = 20
bs = 256
enc_dimension = 60
shuffle = True
for user in users_list:
    user.train(ep, bs, shuffle, enc_dimension)

#check for overfitting
plt.figure()
plt.plot(range(len(users_list[0].t_hist.history["loss"])), users_list[0].t_hist.history["loss"], label = 'training loss')
plt.plot(range(len(users_list[0].t_hist.history["val_loss"])), users_list[0].t_hist.history["val_loss"], label = 'validation loss')
plt.xlabel("epochs")
plt.ylabel("mse")
plt.legend()

for user in users_list:
    user.dec = user.model.predict(user.test_data)

plot_two_CSI_curves(users_list[0].test_data, users_list[0].dec, ant = 0, t = 'CSI for 1 subcarrier') 





# c = np.where((user_pos[1:,0] - user_pos[0:-1,0]) < 5)[0]
# d = np.where(np.abs(user_pos[1:,1] - user_pos[0:-1,1]) != 5)[0]

    




# x_train = []
# x_test = []
# for i in range(size):
#     if i < size - int(size / 5):
#         x_train.append(pre_process(np.load(samples_path+filename+n+'.npy')))
#     else:
#         x_test.append(pre_process(np.load(samples_path+filename+n+'.npy')))
#     n = n[0:len(n) - len(str(i))] + str(i)

# x_train_2 = []
# x_test_2 = []
    
# for i in range(size):
#     if i < size - int(size / 5):
#         x_train_2.append(pre_process(np.load(samples_path+filename+n+'.npy')))
#     else:
#         x_test_2.append(pre_process(np.load(samples_path+filename+n+'.npy')))
#     n = n[0:len(n) - len(str(i))] + str(i)
    


# x_train = np.array(x_train)
# x_test = np.array(x_test)


# model = Sequential()
# model.add(keras.Input(shape=(6400, )))
# model.add(layers.Dense(2000, activation = 'relu', name = 'h1'))
# model.add(layers.Dense(6400, activation = 'sigmoid', name = 'out'))
# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# hist = model.fit(x_train, x_train, validation_split = 0.2, epochs = 10, batch_size = 32)

# plt.figure(1)
# plt.title('Training history')
# plt.plot(range(len(hist.history['loss'])), hist.history['loss'], label = 'training loss')
# plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], label = 'validation loss')
# plt.legend()


# decs = model.predict(x_test)

# decs = decs.reshape(decs.shape[0], 64, 100)
# x_test = x_test.reshape(x_test.shape[0], 64, 100)

# user = 10
# ant = 0
# plot_two_CSI_curves(x_test[user][ant], decs[user][ant], 
#                 t = 'CSI signal\nantenna position: x = ' + str(ant_pos[ant][0]) + ', y = '+ str(ant_pos[ant][1]) + '\nuser position: x = ' + str(user_pos[user][0]) + ', y = ' + str(user_pos[user][1]), 
#                 labelx = 'x_mean = '+str(np.mean(x_test[user][ant]).real), labely = 'y_mean = ' + str(np.mean(decs[user][ant])))