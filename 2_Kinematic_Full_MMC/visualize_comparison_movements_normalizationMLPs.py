"""
Visualization of the MMC network movement performance
when using different MLPs for normalization of the segment vectors.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import pickle

###########################################
# Loading the training data and structure #
###########################################
# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]
# 2) Next level list: two histories = [distance to target, velocity end effector] 
# 3) multiple test movements for each architectur, n=21*20 =420
# 4) and as entries on next level the associated time series (100 time steps)
# Loading the movement data from the pickle file
with open('Results/movements_bw_points_NNnorm', 'rb') as file_pi:
    hist_list = pickle.load(file_pi) 

# As a baseline: loading data when using Euclidean norm for normalization of segment vectors
with open('Results/movements_bw_points_linalgnorm', 'rb') as file_pi:
    hist_norm = pickle.load(file_pi) 
linalg_norm_mean = np.mean(hist_norm[0], axis=0)
linalg_norm_std = np.std(hist_norm[0], axis=0)
linalg_norm_lower_std = linalg_norm_mean - linalg_norm_std
linalg_norm_upper_std = linalg_norm_mean + linalg_norm_std 

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

# Construct data at end of training
copied_arch_last_dist = np.zeros((len(hist_list), len(hist_list[0][0])))
for arch_n in range(0,len(hist_list)):
    for diff_runs in range(0, len(hist_list[arch_n][0])):
        # Getting the last loss - at end of training
        copied_arch_last_dist[arch_n][diff_runs] = hist_list[arch_n][0][diff_runs][-1]
mean_arch_last_dist = np.mean(copied_arch_last_dist, axis=1)
std_arch_last_dist = np.std(copied_arch_last_dist, axis=1)
arch_val_last_dist_lower_std = mean_arch_last_dist - std_arch_last_dist
arch_val_last_dist_upper_std = mean_arch_last_dist + std_arch_last_dist        
#print(mean_arch_last_dist, arch_val_last_dist_lower_std, arch_val_last_dist_upper_std)

dist_run = np.zeros([len(hist_list), len(hist_list[0][0]), len(hist_list[0][0][0])])
for arch_n in range(0,len(hist_list)):
    for diff_runs in range(0, len(hist_list[arch_n][0])):
        for iter in range(0, len(hist_list[arch_n][0][0])):
            dist_run[arch_n][diff_runs][iter] = hist_list[arch_n][0][diff_runs][iter]
mean_dist_run = np.mean(dist_run, axis=1)
std_dist_run = np.std(dist_run, axis=1)
arch_dist_run_lower_std = mean_dist_run - std_dist_run
arch_dist_run_upper_std = mean_dist_run + std_dist_run
#print(mean_arch_last_dist, arch_val_last_dist_lower_std, arch_val_last_dist_upper_std)

###########################################################
# 1 - Comparison different Architectures at end of Movement #
###########################################################
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
ax_arch.spines['left'].set_position('zero')
#ax_arch.set_yscale('log')
ax_arch.set_xlim(-2, 11.5)  
ax_arch.set_xticks(np.arange(-1,len(hist_list)))
ax_arch.set_xticklabels(['Norm','2','4','8','16','32','64','128','[2,2]','[4,4]','[8,8]','[16,16]','[32,32]'])

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_arch_last_dist)), arch_val_last_dist_lower_std,  
                 arch_val_last_dist_upper_std, color=tableau20[1], alpha=0.5) 
plt.errorbar([-1.], [linalg_norm_mean[-1]], [linalg_norm_std[-1]], marker='o', capsize=3, color=tableau20[2])
plt.plot([-2, 11], [linalg_norm_mean[-1], linalg_norm_mean[-1]], linestyle=':', color=tableau20[2])
plt.plot(range(0,len(mean_arch_last_dist)), mean_arch_last_dist, color=tableau20[0], lw=2)
#plt.plot([-1,12], [0.119, 0.119], '--', color=tableau20[6], lw=2) #Squared error from Regression

ax_arch.set_xlabel('# Hidden units', fontsize=14)
ax_arch.set_ylabel('Normalized Mean Distance', fontsize=14)
#ax_arch.set_title('MSE after Learning', fontsize=20)   
plt.savefig("Results/Fig_MeanLastDistance_normalizationMLPs.pdf")

#####################################################
# 2 - Draw figure distance to target over time #
#####################################################
# Which MLPs should be visualized - provide indices in a list (here third arch. and 10th).
exp_hidden_n = [3,10]
arch = [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]

fig = plt.figure(figsize=(8, 6))
# # Remove the plot frame lines. They are unnecessary chartjunk.  
ax_movement = plt.subplot(111)  
#ax_movement.set_title('Distance to target over time for ' + str(arch[exp_hidden_n[0]]) + ' hidden units')
ax_movement.spines["top"].set_visible(False)  
ax_movement.spines["right"].set_visible(False)  
 
# As a reference: Results for linalg normalization
# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(linalg_norm_mean)), linalg_norm_lower_std,  
    linalg_norm_upper_std, color=tableau20[3], alpha=0.5) 
plt.plot(range(0,len(linalg_norm_mean)), linalg_norm_mean, color=tableau20[2], lw=1)
 
color_count = 0
# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_dist_run[exp_hidden_n[0]])), arch_dist_run_lower_std[exp_hidden_n[0]],  
                 arch_dist_run_upper_std[exp_hidden_n[0]], color=tableau20[color_count*2 + 1], alpha=0.5)
for exp_n in exp_hidden_n: 
    plt.plot(range(0,len(mean_dist_run[exp_n])), mean_dist_run[exp_n], color=tableau20[color_count*2], lw=1)
    color_count += 2
# plt.plot(range(0,len(mean_val_loss)), mean_val_loss, color=tableau20[0], lw=1)
# 
ax_movement.set_xlabel('Iteration', fontsize=14)
ax_movement.set_ylabel('Normalized Mean Distance', fontsize=14)
#ax_general.set_title('MSE over Learning for 128 Hidden Neurons', fontsize=20)   
plt.savefig("Results/Fig_MeanMovements_normalizationMLPs.pdf")
plt.show()
