"""
Compute normal equation for normalization task.

This provides a baseline as a comparison for the NN-based approaches.
"""
import numpy as np
import matplotlib.pyplot as plt

######################################
# Generate training data #############
######################################
# Input data: Vectors with a uniformly distributed orientation and 
#             length uniformly distributed from ]0, 2]
angle_steps = 1440
generate_angles = np.arange((-np.pi + 2*np.pi/angle_steps), (np.pi + 2*np.pi/angle_steps),2*np.pi/angle_steps)
input_data = np.empty((angle_steps,2))
target_data = np.empty((angle_steps,2))
for i in range(0, len(generate_angles)):
    target_data[i][0] = np.cos(generate_angles[i])
    target_data[i][1] = np.sin(generate_angles[i])
    
    rand_fact = 2*np.random.random()
    input_data[i][0] = rand_fact * target_data[i][0]
    input_data[i][1] = rand_fact *target_data[i][1]

indices = np.random.permutation(input_data.shape[0])
training_idx, test_idx = indices[:int(0.8 * input_data.shape[0])], indices[int(0.8 * input_data.shape[0]):]
# Construct training and test set
X_train = input_data[training_idx,:]
X_test = input_data[test_idx, :]

Targets_train = target_data[training_idx, :]
Targets_test = target_data[test_idx, :]

X_train_mat = np.mat(X_train)
Targets_train_mat = np.mat(Targets_train)

# Calculate Pseudoinverse for solving normal equation
w_normal = np.linalg.inv(X_train_mat.T * X_train_mat) * X_train_mat.T * Targets_train_mat

mse_test = np.mean( np.square((X_test * w_normal - Targets_test)) ) 
mse_test_angles = np.mean(np.square((X_test * w_normal - Targets_test)), axis=0)
mse_train = np.mean(np.linalg.norm( (X_train * w_normal - Targets_train), axis = 1 ) )
print(mse_test, mse_test_angles)