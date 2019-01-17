"""
Comparing different MMC networks:

The dynamic extension of the network is analyzed: 
the Fig. gives an overview of the characteristic temporal behavior of MMC network types. 

The MMC network was extended by dynamic equations and normalization networks with 
two hidden layers each consisting of 16 neurons were employed. 
A velocity damping factor of 5 was used (damping factor for the other equations was kept at d=10). 

420 reaching movements to targets uniformly distributed on three half circles were performed.
Performance was accessed as distance between end effector and target over time. 
In the early phase, the dynamic MMC lagged the classical approach, but caught up after 
around 10 iterations. Both networks reached a similar level after 40 iterations. 

While the classical MMC approach shows an initial high peaked velocity, 
the dynamic MMC network reached a much lower peak velocity and shows a nice bell-shaped 
velocity profile as characteristic for biological motion.

"""
import numpy as np
import matplotlib.pyplot as plt
from mmcDynamicArmNetwork import mmcDynamicArmNetwork
import sys
sys.path.insert(0, '../2_Kinematic_Full_MMC/')
from mmcArmNetwork import mmcArmNetwork

import pickle

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

hist_list = []

# Setting up the targets on three half circles.
target_points_on_circle = 6
targets = []
for radius in range(1, 4):
    for step in range(0, target_points_on_circle+1):
        targets.append(np.array([radius * np.cos(step*(np.pi/target_points_on_circle)), 
            radius * np.sin(step*(np.pi/target_points_on_circle)) ] ))

# Dynamic MMC Network - Experiment
########################################
mmc_arm = mmcDynamicArmNetwork('normalization_net_xy_inOut_[16, 16].h5')
#mmc_arm.initialise_drawing_window()

hist_dist_dyn = []
hist_vel_dyn = []
for start_target in targets:
    print("Next start target: ", start_target)
    for end_target in targets:
        if (np.linalg.norm(start_target - end_target) > 0.01):
            mmc_arm.reset_all_variables()
            mmc_arm.set_new_target(start_target)
            mmc_arm.mmc_iteration_step(40)
            mmc_arm.set_new_target(end_target)
            dist, vel = mmc_arm.mmc_iteration_step(40)
            hist_dist_dyn.append(dist)
            hist_vel_dyn.append(vel)
 
dist_dyn_mean = np.mean(np.array(hist_dist_dyn), axis=0)
dist_dyn_std = np.std(np.array(hist_dist_dyn), axis=0)
dist_dyn_lower = dist_dyn_mean - dist_dyn_std
dist_dyn_upper = dist_dyn_mean + dist_dyn_std
vel_dyn_mean = np.mean(np.array(hist_vel_dyn), axis=0)
vel_dyn_std = np.std(np.array(hist_vel_dyn), axis=0)
vel_dyn_lower = vel_dyn_mean - vel_dyn_std
vel_dyn_upper = vel_dyn_mean + vel_dyn_std

# Classical MMC Network as a Baseline
########################################
mmc_arm = mmcArmNetwork()           
hist_dist = []
hist_vel = []
for start_target in targets:
    print("Next start target: ", start_target)
    for end_target in targets:
        if (np.linalg.norm(start_target - end_target) > 0.01):
            mmc_arm.reset_all_variables()
            mmc_arm.set_new_target(start_target)
            mmc_arm.mmc_iteration_step(40)
            mmc_arm.set_new_target(end_target)
            dist, vel = mmc_arm.mmc_iteration_step(40)
            hist_dist.append(dist)
            hist_vel.append(vel)

dist_kin_mean = np.mean(np.array(hist_dist), axis=0)
dist_kin_std = np.std(np.array(hist_dist), axis=0)
dist_kin_lower = dist_kin_mean - dist_kin_std
dist_kin_upper = dist_kin_mean + dist_kin_std

vel_kin_mean = np.mean(np.array(hist_vel), axis=0)
vel_kin_std = np.std(np.array(hist_vel), axis=0)
vel_kin_lower = vel_kin_mean - vel_kin_std
vel_kin_upper = vel_kin_mean + vel_kin_std

#linalg_norm_std = np.std(hist_norm[0], axis=0)
#linalg_norm_lower_std = linalg_norm_mean - linalg_norm_std
#linalg_norm_upper_std = linalg_norm_mean + linalg_norm_std

# Visualizing difference between Approaches
########################################
fig = plt.figure(figsize=(6, 6))
ax_movement = plt.subplot(111)  
#ax_movement.set_title('Distance to target over time for ' + str(arch[exp_hidden_n[0]]) + ' hidden units')
ax_movement.set_xlim([0,40])
ax_movement.spines["top"].set_visible(False)  
ax_movement.spines["right"].set_visible(False)  

# Use matplotlib's fill_between() call to create error bars.    
plt.plot(range(0,len(dist_kin_mean)), dist_kin_mean, color=tableau20[2], lw=1)
plt.fill_between(range(0,len(dist_kin_mean)), dist_kin_upper,  
                 dist_kin_lower, color=tableau20[3], alpha=0.5)
plt.fill_between(range(0,len(dist_dyn_mean)), dist_dyn_upper,  
                 dist_dyn_lower, color=tableau20[5], alpha=0.5)
plt.plot(range(0,len(dist_dyn_mean)), dist_dyn_mean, color=tableau20[4], lw=1)
#plt.plot(range(0,len(vel_dyn_mean)), vel_dyn_mean, color=tableau20[5], lw=1)
#plt.plot(range(0,len(vel_kin_mean)), vel_kin_mean, color=tableau20[7], lw=1)
ax_movement.set_xlabel('Iteration', fontsize=14)
ax_movement.set_ylabel('Normalized Mean Distance', fontsize=14)
#ax_general.set_title('MSE over Learning for 128 Hidden Neurons', fontsize=20)   
plt.savefig("Results/Fig_MeanDistance_DynamicComp.pdf")

fig = plt.figure(figsize=(6, 6))
ax_velocity = plt.subplot(111)  
#ax_movement.set_title('Distance to target over time for ' + str(arch[exp_hidden_n[0]]) + ' hidden units')
ax_velocity.set_xlim([0,40])
ax_velocity.spines["top"].set_visible(False)  
ax_velocity.spines["right"].set_visible(False)  

# Use matplotlib's fill_between() call to create error bars.    
plt.plot(range(0,len(vel_kin_mean)), vel_kin_mean, color=tableau20[2], lw=1)
plt.fill_between(range(0,len(vel_kin_mean)), vel_kin_upper,  
                 vel_kin_lower, color=tableau20[3], alpha=0.5)
plt.fill_between(range(0,len(vel_dyn_mean)), vel_dyn_lower,  
                 vel_dyn_upper, color=tableau20[5], alpha=0.5)
plt.plot(range(0,len(vel_dyn_mean)), vel_dyn_mean, color=tableau20[4], lw=1)
#plt.plot(range(0,len(vel_dyn_mean)), vel_dyn_mean, color=tableau20[5], lw=1)

#plt.plot(range(0,len(vel_kin_mean)), vel_kin_mean, color=tableau20[7], lw=1)
ax_velocity.set_xlabel('Iteration', fontsize=14)
ax_velocity.set_ylabel('Mean Velocity', fontsize=14)
#ax_general.set_title('MSE over Learning for 128 Hidden Neurons', fontsize=20)   
plt.savefig("Results/Fig_MeanVelocity_DynamicComp.pdf")
plt.show()


        