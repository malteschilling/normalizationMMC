"""
Visualizing a hidden layer of a MLP.

The MLP is trained on a normalization function from a vector input with the goal of 
normalizing the noisy input vector.
"""
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

######################################
# Parameters ######################### 
######################################
batch_size = 16
epochs = 400
hidden_size = 8
angle_steps = 3600
l2_regular_weight = 0.0005

######################################
# Generate training data #############
######################################
# Input data: Vectors with a uniformly distributed orientation and 
#             length uniformly distributed from ]0, 2]
generate_angles = np.arange((-np.pi + 2*np.pi/angle_steps), (np.pi + 2*np.pi/angle_steps),2*np.pi/angle_steps)
#target_norm = (target_angles/(2*np.pi)) + 0.5
input_data = np.empty((angle_steps,3))
target_data = np.empty((angle_steps,3))
for i in range(0, len(generate_angles)):
    target_data[i][0] = np.cos(generate_angles[i])
    target_data[i][1] = np.sin(generate_angles[i])
    target_data[i][2] = generate_angles[i]/np.pi
    
    rand_fact = 2*np.random.random()
    input_data[i][0] = rand_fact * target_data[i][0]
    input_data[i][1] = rand_fact *target_data[i][1]
    input_data[i][2] = target_data[i][2]

indices = np.random.permutation(input_data.shape[0])
training_idx, test_idx = indices[:int(0.8 * input_data.shape[0])], indices[int(0.8 * input_data.shape[0]):]
# Construct training and test set
#training_data, test_data = all_data_set[training_idx,:], all_data_set[test_idx,:]
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
model = Sequential()
model.add(Dense(hidden_size, activation='tanh', input_dim=2, kernel_regularizer=regularizers.l2(l2_regular_weight), name="dense_one"))# input_shape=(3,)))

model.add(Dense(hidden_size, activation='tanh', input_dim=2, kernel_regularizer=regularizers.l2(l2_regular_weight), name="dense_two"))# input_shape=(3,)))
model.add(Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(l2_regular_weight)))
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

#model.save('normalization_net_xy_inOut_16.h5')  # creates a HDF5 file 'my_model.h5'

intermediate_layer_model_one = Model(inputs=model.input,
                                 outputs=model.get_layer("dense_one").output)
intermediate_layer_model_one.compile(loss='mean_squared_error', #metrics=[r_square],
                  optimizer=Adam())
                  
# intermediate_layer_model_two = Model(inputs=model.input,
#                                  outputs=model.get_layer("dense_two").output)
# intermediate_layer_model_two.compile(loss='mean_squared_error', #metrics=[r_square],
#                   optimizer=Adam())
# intermediate_output = intermediate_layer_model_one.predict(Xnew)
# print("Intermediate output: ", intermediate_output)
# intermediate_output = intermediate_layer_model_two.predict(Xnew)
# print("Intermediate output: ", intermediate_output)

hidden_activ_one = np.zeros((angle_steps,hidden_size))
hidden_activ_two = np.zeros((angle_steps,hidden_size))
input_to_hidden = np.zeros((angle_steps,))
for i in range(0, angle_steps):
    intermediate_output_one = intermediate_layer_model_one.predict(np.array([target_data[i,:2]]))
    input_to_hidden[i] = target_data[i,0]
    hidden_activ_one[i] = intermediate_output_one[0]
    
#    intermediate_output_two = intermediate_layer_model_two.predict(np.array([target_data[i,:2]]))
#    hidden_activ_two[i] = intermediate_output_two[0]
#print(target_data)

# Plot activation first hidden layer
fig = plt.figure(figsize=(8, 6))
ax_hidden_act_1 = plt.subplot(111)  
#ax.set_yticklabels(['No Hidden','1','2','4','8','16','32','64','128'])
plt.plot(range(0,len(hidden_activ_one)), hidden_activ_one[:,0], color=tableau20[0])
plt.plot(range(0,len(hidden_activ_one)), hidden_activ_one[:,1], color=tableau20[2])
plt.plot(range(0,len(hidden_activ_one)), hidden_activ_one[:,2], color=tableau20[4])
plt.plot(range(0,len(hidden_activ_one)), hidden_activ_one[:,3], color=tableau20[6])
ax_hidden_act_1.set_xlabel('Vector Angle (degree)', fontsize=14)
ax_hidden_act_1.set_ylabel('Activation', fontsize=14)
ax_hidden_act_1.set_title('Activation First Hidden Layer', fontsize=20) 
plt.savefig("Results/Fig_Hidden_ActivationHiddenLayer.pdf")

fig_3D = plt.figure(figsize=(10, 8))
vis_angle_steps = np.arange(0, angle_steps)
# Number of variations of architectures
vis_hidden_size = np.arange(0, hidden_size)
vis_X, vis_Y = np.meshgrid(vis_angle_steps, vis_hidden_size)
print(vis_X.shape, hidden_activ_one.shape)
vis_hidden_activ = np.zeros(vis_X.shape)

# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_3D = plt.subplot(111, projection='3d')   
for i in range(0, hidden_size):
    ax_3D.plot(vis_angle_steps/10., ys=np.ones(angle_steps)*i, zs=hidden_activ_one[:,i], zdir='z', label='curve in (x,y)')

ax_3D.set_xlabel('Vector Angle (degree)', fontsize=14)
ax_3D.set_ylabel('Hidden Unit Number', fontsize=14, labelpad=12)
ax_3D.set_zlabel('Activation', fontsize=14)

plt.savefig("Results/Fig_Hidden_ActivationHiddenLayer_3D.pdf")
plt.show()
