"""
Comparing different MMC networks: 

Integrated are the sub-networks for normalization - this means that the
normalization step of the MMC network is realized as an MLP.

Here, different MLP complexities are compared for a series of movements.
The MMC network was tested in a series of inverse kinematic tasks:
21 points were distributed through half of the whole working space in a systematic manner.
3 half-circles around the base of the manipulator were used to arrange 7 points on each 
of these half-circles (radius of 1, 2 and 3 units).
Overall, this lead to 21 target points and the goal of the MMC network was to make 
reaching movements starting once from every one of these points towards each of the 
other points. This resulted in 420 reaching movements.

Movement of the arm was recorded over all movements. 

Visualization shows normalized distance over time towards the target point. 
The distance is normalized with respect to the distance from start to target point. 

The comparison shows that the network qualitatively behaves similar to the classical MMC 
approach and the distance continuously decreased. For the selected complexity of the 
normalization sub-net the difference between reached and target position is somewhat higher. 
Standard deviation of the distance is given as the shaded area. 

Overall, the large variance is due to the fact that many (quite diverse) movements 
are pooled together from quite different arm configurations. 

Importantly, both curves show the same trend and the movement profile nicely visualizes 
the main drawback from classical MMC networks: the distance decreases exponentially.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmcArmNetwork import mmcArmNetwork

import pickle

target_points_on_circle = 6

hist_list = []

# Setting up the targets on three half circles.
targets = []
for radius in range(1, 4):
    for step in range(0, target_points_on_circle+1):
        targets.append(np.array([radius * np.cos(step*(np.pi/target_points_on_circle)), 
            radius * np.sin(step*(np.pi/target_points_on_circle)) ] ))

# Loading the MLPs with different complexities.
hidden_size = [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]
for hidd_size in hidden_size:
    print("*************************")
    print("HIDDEN SIZE: ", hidd_size)
    print("*************************")
    model_name = 'normalization_net_xy_inOut_' + str(hidd_size) + '.h5'
    mmc_arm = mmcArmNetwork(norm_model_name=model_name)
    hist_dist = []
    hist_vel = []
    #  mmc_arm.initialise_drawing_window()
    for start_target in targets:
        print("Next start target: ", start_target)
        for end_target in targets:
            if (np.linalg.norm(start_target - end_target) > 0.01):
                mmc_arm.reset_all_variables()
                mmc_arm.set_new_target(start_target)
                mmc_arm.mmc_iteration_step(100)
                mmc_arm.set_new_target(end_target)
                dist, vel = mmc_arm.mmc_iteration_step(100)
                hist_dist.append(dist)
                hist_vel.append(vel)
    hist_list.append([np.array(hist_dist), np.array(hist_vel)])
    print(np.mean(np.array(hist_dist), axis=0))
    
#with open('movements_bw_points_NNnorm', 'wb') as file_pi:
 #   pickle.dump(hist_list, file_pi) 

# Make a movement and show it in an interactive figure
#mmc_arm = mmcArmNetwork()
#mmc_arm.initialise_drawing_window()
#mmc_arm.mmc_iteration_step(50)
#mmc_arm.set_new_target(np.array([3.,0.]))
#mmc_arm.mmc_iteration_step(50)
#mmc_arm.plot_velocity_distance()
        