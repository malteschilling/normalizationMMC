"""
Compare the different trained MLPs for normalization of vectors 
(input vector have a length in ]0, 2], output is unit vector with same orientation).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as py
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
# 2) Next level list: multiple training runs from random initializations, 5
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file
with open('Results/trainHistoryDict_5runs_400ep_normalizeVector', 'rb') as file_pi:
    hist_list = pickle.load(file_pi) 
    
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

##########################################
# 1A - Construct Data for 3D surface plot #
##########################################
# Construct target arrays for the 3D surface visualisation data
vis_epochs = np.arange(0, len(hist_list[0][0]['val_loss']))
# Number of variations of architectures
vis_hid_var = np.arange(0,len(hist_list))
# Construct meshgrid of the two different arrays
# = three 2 dimensional arrays, vis_X and vis_Y define regular locations in two dimensional space
#   while vis_val_loss provides the specific values
vis_X, vis_Y = np.meshgrid(vis_epochs, vis_hid_var)
vis_val_loss = np.zeros(vis_X.shape)

# Pushing the loaded data into this grid structure (vis_val_loss)
# For this: the mean over the multiple runs for a single architecture is first calculated
for arch_n in range(0,len(hist_list)):
    copied_val_loss = np.zeros((len(hist_list[arch_n]), len(hist_list[arch_n][0]['val_loss']) ))
    for diff_runs in range(0, len(hist_list[arch_n])):
        copied_val_loss[diff_runs] = hist_list[arch_n][diff_runs]['val_loss']
    vis_val_loss[arch_n] = np.log(np.mean(copied_val_loss, axis=0)) # np.log(np.array(hist_list[i]['val_loss']))
    #vis_val_loss[i][vis_val_loss[i]>20]= 20
    
# Construct data at end of training
copied_arch_val_loss = np.zeros((len(hist_list), len(hist_list[0])))
for arch_n in range(0,len(hist_list)):
    for diff_runs in range(0, len(hist_list[arch_n])):
        # Getting the last loss - at end of training
        copied_arch_val_loss[arch_n][diff_runs] = hist_list[arch_n][diff_runs]['val_loss'][-1]
mean_arch_val_loss = np.mean(copied_arch_val_loss, axis=1)
print(copied_arch_val_loss[2])
std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
arch_val_loss_lower_std = mean_arch_val_loss - std_arch_val_loss
arch_val_loss_upper_std = mean_arch_val_loss + std_arch_val_loss        
################################################
# 1 B - Draw 3D figure for the error over time #
################################################
#py.ion()
fig = plt.figure(figsize=(10, 8))
     
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_3D = plt.subplot(111, projection='3d')   
# Make sure your axis ticks are large enough to be easily read.  
# You don't want your viewers squinting to read your plot.  
#plt.xticks(fontsize=14)  
#plt.yticks(fontsize=14)  
ax_3D.set_yticklabels(['2','4','8','16','32','64','128','[2,2]','[4,4]','[8,8]','[16,16]','[32,32]'])

plt.ylim(0, 12)    
surf = ax_3D.plot_surface(vis_X, vis_Y, vis_val_loss, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
plt.gca().invert_yaxis()
#ax.set_zscale('log')
fig.colorbar(surf, shrink=0.5, aspect=5)

ax_3D.set_xlabel('Epoch', fontsize=14)
ax_3D.set_ylabel('# Hidden Units', fontsize=14)
ax_3D.set_zlabel('log MSE', fontsize=14)
#ax_3D.set_title('MSE over Learning for NN Architectures', fontsize=20)   
py.savefig("Results/Fig_Comparison_3D_MLPs_MSE.pdf")

###########################################################
# 1 C - Comparison different Architectures after Training #
###########################################################
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
     
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 12)  
ax_arch.set_xticks(np.arange(0,len(hist_list)))
ax_arch.set_xticklabels(['2','4','8','16','32','64','128','[2,2]','[4,4]','[8,8]','[16,16]','[32,32]'])

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_arch_val_loss)), arch_val_loss_lower_std,  
                 arch_val_loss_upper_std, color=tableau20[1], alpha=0.5) 

plt.plot(range(0,len(mean_arch_val_loss)), mean_arch_val_loss, color=tableau20[0], lw=2)
plt.plot([-1,12], [0.073, 0.073], '--', color=tableau20[6], lw=2) #Squared error from Regression

ax_arch.set_xlabel('# Hidden units', fontsize=14)
ax_arch.set_ylabel('MSE', fontsize=14)
#ax_arch.set_title('MSE after Learning', fontsize=20)   
py.savefig("Results/Fig_Comparison_MLPs_MSE.pdf")

###############################################
# 2 A - Visualize generalization for large NN #
###############################################
# For a specified number of hidden neurons:
# Show mean time series of val_loss and trainings loss
# = illustrates that there is no overfitting
exp_hidden_n = 8
arch = [2,4,8,16,32,64,128,[2,2],[4,4],[8,8],[16,16],[32,32]]
copied_val_loss = np.zeros((len(hist_list[-1]), len(hist_list[-1][0]['val_loss']) ))
for diff_runs in range(0, len(hist_list[-1])):
    copied_val_loss[diff_runs] = hist_list[exp_hidden_n][diff_runs]['val_loss']
mean_val_loss = np.mean(copied_val_loss, axis=0)
std_val_loss = np.std(copied_val_loss, axis=0)
val_loss_lower_std = mean_val_loss - std_val_loss
val_loss_upper_std = mean_val_loss + std_val_loss

copied_loss = np.zeros((len(hist_list[-1]), len(hist_list[-1][0]['loss']) ))
for diff_runs in range(0, len(hist_list[-1])):
    copied_loss[diff_runs] = hist_list[exp_hidden_n][diff_runs]['loss']
mean_loss = np.mean(copied_loss, axis=0)
std_loss = np.std(copied_loss, axis=0)
loss_lower_std = mean_loss - std_loss
loss_upper_std = mean_loss + std_loss

#####################################################
# 2 B - Draw figure showing training and test error #
#####################################################
fig = plt.figure(figsize=(8, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_general = plt.subplot(111)  
ax_general.set_title('Learning over time for ' + str(arch[exp_hidden_n]) + ' hidden units')
ax_general.spines["top"].set_visible(False)  
ax_general.spines["right"].set_visible(False)  
     
ax_general.set_yscale('log')

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_loss)), loss_lower_std,  
                 loss_upper_std, color=tableau20[3], alpha=0.5) 
plt.fill_between(range(0,len(mean_loss)), val_loss_lower_std,  
                 val_loss_upper_std, color=tableau20[1], alpha=0.5) 

plt.plot(range(0,len(mean_loss)), mean_loss, color=tableau20[2], lw=1)
plt.plot(range(0,len(mean_val_loss)), mean_val_loss, color=tableau20[0], lw=1)

ax_general.set_xlabel('Epoch', fontsize=14)
ax_general.set_ylabel('MSE (log)', fontsize=14)
#ax_general.set_title('MSE over Learning for 128 Hidden Neurons', fontsize=20)   
py.savefig("Results/Fig_Comparison_MSE_overTime_TestTraining.pdf")
plt.show()