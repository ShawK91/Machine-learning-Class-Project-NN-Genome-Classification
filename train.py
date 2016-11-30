import numpy as np, sys
from matplotlib import pyplot as plt
plt.switch_backend('Qt4Agg')
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import SReLU
from keras.regularizers import l2, activity_l2
#from keras.optimizers import SGD

import mod_memoried as mod

#Macro Variables
epoch = 1000
num_agents = 3
num_hnode = 100
stat_run = 1
neural_input = 75



def load_model(model, num_agents, foldername = 'Models/'):
    import copy
    all_models = []
    for i in range(num_agents):
        ig = copy.deepcopy(model)
        ig.load_weights(foldername + 'Agent' + str(i) + '.h5')
        all_models.append(ig)
        all_models[i].compile(loss='mse', optimizer='Nadam')
    return all_models

def save_model(model, agent_index, foldername = '/Models/'):
    import os
    #Create folder to store all networks if not present
    filename = os.getcwd() + foldername
    if not os.path.exists(os.path.dirname(filename)):
        try: os.makedirs(os.path.dirname(filename))
        except: 1+1
    #Save weights
    model.save_weights('Models/Agent' + str(agent_index) + '.h5', overwrite=True)



def init_model(num_input, num_hnodes, num_output):


    model = Sequential()
    model.add(Dense(num_hnodes, input_dim=num_input, init='he_uniform', W_regularizer=l2(0.01)))
    #model.add(Activation('sigmoid'))
    model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    model.add(Dense(num_output,init='he_uniform'))
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=True)
    nadam = keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='mean_absolute_error', optimizer=nadam)
    return model

def data_preprocess(foldername = 'Training Data/'):
     all_data = []
     #Import training data
     for agent in range(num_agents):
        ig_x = np.loadtxt(foldername + str(agent) + '/train_x.csv')
        ig_x = np.reshape(ig_x, (ig_x.shape[0], ig_x.shape[1]))
        #x.append(ig_x)
        ig_y = np.loadtxt(foldername + str(agent) + '/train_y.csv')
        ig_y = np.reshape(ig_y, (ig_y.shape[0], 1))
        #y.append(ig_y)
        ig = np.concatenate((ig_x, ig_y), axis=1)
        np.random.shuffle(ig)
        all_data.append(ig)

     # Figure out the split
     num_examples  = len(all_data[0])
     train_end_ind = int(0.6 * num_examples)
     val_end_ind = int(0.75 * num_examples)

     #Create validation and test sets
     train_data = []; val_data = []; test_data = []
     for agent in range(num_agents):
         train_data.append(all_data[agent][0:train_end_ind])
         val_data.append(all_data[agent][train_end_ind:val_end_ind])
         test_data.append(all_data[agent][val_end_ind:])

     return train_data, val_data, test_data

def main():
    #Load data
    train_data, val_data, test_data = data_preprocess()
    #load_models = load_model(all_models[0], num_agents)

    ##################### TRAIN SIMULATOR ##########################
    all_models = []; history = []
    for agent in range(num_agents):
        print 'Training Agent ', agent
        all_models.append(init_model(num_input=neural_input, num_hnodes=num_hnode, num_output=1))
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        history.append(all_models[agent].fit(train_data[agent][:,:-1], train_data[agent][:,-1:], nb_epoch=epoch, batch_size=16, verbose=1, callbacks=[early_stopping], shuffle=True, validation_data=(val_data[agent][:,:-1], val_data[agent][:,-1:])))

    #Save training history to csv
    for agent in range(num_agents):
        # Get metrics from dictionary
        train_loss = history[agent].history['loss']
        val_loss = history[agent].history['val_loss']

        # Save metrics
        decorator = np.arange(len(train_loss))
        decorator = np.reshape(decorator, (len(decorator), 1))
        train_loss = np.array(train_loss)
        train_loss = np.reshape(train_loss, (len(train_loss),1))
        val_loss = np.array(val_loss)
        val_loss = np.reshape(val_loss, (len(val_loss),1))

        train_loss_save = np.concatenate((decorator, train_loss), axis=1)
        val_loss_save = np.concatenate((decorator, val_loss), axis=1)

        np.savetxt('Agent_' + str(agent) +'_train_loss.csv', train_loss_save, delimiter=',')
        np.savetxt('Agent_' + str(agent) + '_val_loss.csv', val_loss_save, delimiter=',')




    #Save trained models
    for index, model in enumerate(all_models):
        save_model(model, index)





    ##################### TEST SIMULATOR ##########################






















if __name__ == '__main__':
    main()
