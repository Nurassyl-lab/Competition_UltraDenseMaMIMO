"Ultra Dense Indoor MaMIMO CSI Dataset"
from competition_utils import *

#inital variables
samples_path = 'DIS_lab_LoS/samples/'
numb_users = 4#corresponds to number of models
size = int(252004 / numb_users) #size of user dataset
select_ant = False

#create list of users
users_list = [USER(i+1) for i in range(numb_users)]

#load data
size = 10000
print("Loading data from ("+samples_path+") folder...")
n = '000000'
antennas = np.array([i for i in range(64)])#64 antennas in total
antennas = antennas.reshape(numb_users, int(64 / numb_users))
pref_ant = 0
for i in range(len(user_pos)):
    if select_ant:#it will assign sample to a closest stack of antennas
        user_x = user_pos[i][0]
        user_y = user_pos[i][1]
        closest = 999999
        
        for j in range(len(ant_pos)):
            ant_x = ant_pos[j][0]
            ant_y = ant_pos[j][1]
            distance = np.sqrt((np.abs(user_x - ant_x))**2 + (np.abs(user_y - ant_y))**2)
            if distance < closest:
                closest = distance
                pref_ant = j
        #for current user position prefered antenna stack is 
        if len(users_list[np.where(antennas == pref_ant)[0][0]].x) < size:
            tmp = np.load(samples_path+filename+n+'.npy')
            users_list[np.where(antennas == pref_ant)[0][0]].x.append(np.array([tmp.real, tmp.imag]))
    else:
        if len(users_list[np.where(antennas == pref_ant)[0][0]].x)<size:
            tmp = np.load(samples_path+filename+n+'.npy')
            users_list[np.where(antennas == pref_ant)[0][0]].x.append(np.array([tmp.real, tmp.imag]))
        else:
            if pref_ant != 63:
                pref_ant += 1
    n = n[0:len(n) - len(str(i))] + str(i)

#reshape data
for user in users_list:
    user.x = np.array(user.x).reshape(len(user.x), 2, 64, 100, 1)

#train and predict
for user in users_list:
    seed(user.number)
    tensorflow.random.set_seed(user.number)
    print('Training user', user.number)
    user.assign_cnn(act_function='tanh')
    user.model.fit(user.x[0:-1000], user.x[0:-1000], epochs = 10, batch_size = 64)
    print('Decoding...')
    user.dec = user.model.predict(user.x[-1000:])

#evaluate
for user in users_list:
    seed(user.number)
    tensorflow.random.set_seed(user.number)
    evals = user.model.evaluate(user.x[-1000:], user.x[-1000:])
    
    with open("competition/evaluation_record.txt", 'a') as f:
        now = datetime.now()
        f.write(str(now.strftime("%d/%m/%Y %H:%M:%S"))  + ' evalution of user '+str(user.number)+' = '+str(evals) + '\n')





















