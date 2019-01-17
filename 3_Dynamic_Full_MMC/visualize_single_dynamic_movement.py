"""
Visualization of a single movement
- comparison of the classical MMC approach and the fully neural dynamic MMC approach.
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

targets = [np.array([-1.8, 1.8]), np.array([2.,0.7])]

# Dynamic MMC Network - Example Movement
########################################
mmc_arm = mmcDynamicArmNetwork('normalization_net_xy_inOut_[16, 16].h5')
mmc_arm.vel_decay_direct = 1.0

hist_dist_dyn = []
hist_vel_dyn = []
end_target = np.array([-1.8, 1.8])
mmc_arm.reset_all_variables()
mmc_arm.set_new_target(end_target)
dist_dyn, vel_dyn = mmc_arm.mmc_iteration_step(40)
# 0.030 fuer overshoot
# 0.025 fuer direct 0.92

mmc_arm = mmcDynamicArmNetwork('normalization_net_xy_inOut_[16, 16].h5')
mmc_arm.initialise_drawing_window()
mmc_arm.reset_all_variables()
mmc_arm.set_new_target(end_target)
dist_dyn_decay, vel_dyn_decay = mmc_arm.mmc_iteration_step(40)

# For comparison: classical MMC Movement
########################################
mmc_arm = mmcArmNetwork()    
mmc_arm.reset_all_variables()
mmc_arm.set_new_target(end_target)
dist_class, vel_class = mmc_arm.mmc_iteration_step(40)

# Visualization of two movements
########################################
fig = plt.figure(figsize=(6, 6))
ax_velocity = plt.subplot(111)  
ax_velocity.set_xlim([0,40])
ax_velocity.spines["top"].set_visible(False)  
ax_velocity.spines["right"].set_visible(False)  

# Use matplotlib's fill_between() call to create error bars.    
plt.plot(range(0,len(vel_class)), vel_class, color=tableau20[2], lw=1)
plt.plot(range(0,len(vel_dyn)), vel_dyn, color=tableau20[4], lw=1, linestyle='--')
plt.plot(range(0,len(vel_dyn)), vel_dyn_decay, color=tableau20[4], lw=1)
#plt.plot(range(0,len(vel_dyn_mean)), vel_dyn_mean, color=tableau20[5], lw=1)

#plt.plot(range(0,len(vel_kin_mean)), vel_kin_mean, color=tableau20[7], lw=1)
ax_velocity.set_xlabel('Iteration', fontsize=14)
ax_velocity.set_ylabel('Velocity', fontsize=14)
plt.savefig("Results/Fig_DynamicMovement.pdf")
plt.show()
        