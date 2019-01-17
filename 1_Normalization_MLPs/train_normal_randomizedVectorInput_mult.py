"""
Training a normalization function from a vector input.

MLPs with different complexities are trained - using keras.
Goal is to normaliza an input vector.
"""
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import pickle

######################################
# Parameters ######################### 
######################################
batch_size = 16
epochs = 400
# Number of repetitions for each architecture
run_training = 5
# Different architectures - single number = hidden neurons for single hidden layer
#                           list of numbers = multiple layers consisting of # of neurons
hidden_size = [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]
angle_steps = 3600
#l2_regular_weight = 0.0001
# Store the data in list
hist_list = []

######################################
# Generate training data #############
######################################
# Input data: Vectors with a uniformly distributed orientation and 
#             length uniformly distributed from ]0, 2]
generate_angles = np.arange((-np.pi + 2*np.pi/angle_steps), (np.pi + 2*np.pi/angle_steps),2*np.pi/angle_steps)
input_data = np.empty((angle_steps,3))
target_data = np.empty((angle_steps,3))
for i in range(0, len(generate_angles)):
    target_data[i][0] = np.cos(generate_angles[i])
    target_data[i][1] = np.sin(generate_angles[i])
    # Angles are normalized to [-1,1]
    target_data[i][2] = generate_angles[i]/np.pi
    
    rand_fact = 2*np.random.random()
    input_data[i][0] = rand_fact * target_data[i][0]
    input_data[i][1] = rand_fact *target_data[i][1]
    input_data[i][2] = target_data[i][2]

indices = np.random.permutation(input_data.shape[0])
training_idx, test_idx = indices[:int(0.8 * input_data.shape[0])], indices[int(0.8 * input_data.shape[0]):]
# Construct training and test set
X_train = input_data[training_idx,:]
X_test = input_data[test_idx, :]

Targets_train = target_data[training_idx, :]
Targets_test = target_data[test_idx, :]

model = None
print(X_train.shape, Targets_train.shape, X_test.shape, Targets_test.shape)

######################################
########## TRAIN NETWORKS ############
######################################
######################################
# Vary Size of Hidden Layer ##########
######################################
for hidd_size in hidden_size:
    print(" ######## HIDDEN MODEL ######## ")
    print(" ######## ", hidd_size)
    print(" ######## HIDDEN MODEL ######## ")
    hist_list.append([])
    # Run multiple runs for each architecture, size of hidden units
    for run_it in range(0, run_training):
        print(" ######## Trainings run ######## ")
        print(" ######## ", run_it)
        print(" ######## HIDDEN MODEL  ######## ")
        print(" ######## ", hidd_size)
        model = Sequential()
        
        # As a variation: we might want to use also angles as an input, but then
        # we should only provide some of the otherwise redundant information
        #model.add(Dropout(0.25, input_shape=(3,)))
        
        if isinstance(hidd_size,list):
            model.add(Dense(hidd_size[0], activation='tanh', input_dim=2, name=("hidden_layer_0")))# input_shape=(3,)))
            for i in range(1, len(hidd_size)):
                # When using orientation angles of vectors as inputs as well.
                #model.add(Dense(hidd_size[i], activation='tanh', input_dim=3, kernel_regularizer=regularizers.l2(l2_regular_weight), name=("hidden_layer_" + str(i))))# input_shape=(3,)))
                model.add(Dense(hidd_size[i], activation='tanh', name=("hidden_layer_" + str(i))))# input_shape=(3,)))
        else:
            # When using orientation angles of vectors as inputs as well.
            #model.add(Dense(hidden_size, activation='tanh', input_dim=3, kernel_regularizer=regularizers.l2(l2_regular_weight), name="hidden_layer"))# input_shape=(3,)))
            model.add(Dense(hidd_size, activation='tanh', input_dim=2, name="hidden_layer"))# input_shape=(3,)))
        #model.add(Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(l2_regular_weight)))
        model.add(Dense(2, activation='tanh'))
        model.summary()

        # Use MSE and Adam as an optimizer
        model.compile(loss='mean_squared_error', #metrics=[r_square],
                optimizer=Adam())
                  
        # Start training
        history = model.fit(X_train[:,:2], Targets_train[:,:2],
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test[:,:2], Targets_test[:,:2])) 
        hist_list[-1].append(history.history)

# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]
# 2) Next level list: multiple training runs from random initializations, run_training
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file
print(len(hist_list))
print(len(hist_list[0]))
print((hist_list[2][0]))
with open('trainHistoryDict_5runs_400ep_normalizeVector', 'wb') as file_pi:
    pickle.dump(hist_list, file_pi) 