"""
Visualization of the NN for normalization of a noisy vector.

The MLP subnetworks realize a normalization of an input vector (length in ]0, 2])
towards a unit vector with the same orientation. The change of the input vector
to output vector (calculated as the difference) can be visualized. This leads to
this figure that shows the difference vectors that project onto the unit circle over the 
the input vector space.
"""
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import load_model

normalization_model = load_model('trained_normalization_subnetworks/normalization_net_xy_inOut_16.h5')

X = np.arange(-1.6, 1.75, 0.2)
Y = np.arange(-1.6, 1.75, 0.2)
U, V = np.meshgrid(X, Y)
print(U.shape)
Dir_X = np.zeros(U.shape)
Dir_Y = np.zeros(V.shape)

for x_count in range(0, len(X)):
    for y_count in range(0, len(Y)):
        norm_val = np.squeeze(normalization_model.predict( np.array([[X[x_count], Y[y_count]]]) ) ) 
        #print(X[x_count], Y[y_count], norm_val)
        Dir_X[y_count, x_count] = norm_val[0]-X[x_count]
        Dir_Y[y_count, x_count] = norm_val[1]-Y[y_count]

fig = plt.figure(figsize=(6, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax = plt.subplot() 
circ = plt.Circle((0, 0), radius=1, edgecolor=(1., 187./255, 120./255), facecolor='None', linestyle='--')
ax.add_patch(circ)
q = ax.quiver(X, Y, Dir_X, Dir_Y, color=(31./255, 119./255, 180./255))
#ax.set_title('Transformation of noisy inputs towards the unit-circle')
#ax.quiverkey(q, X=0.3, Y=1.1, U=10,
  #           label='Quiver key, length = 10', labelpos='E')
plt.savefig("Results/Fig_Normalization_to_UnitCircle.pdf")
plt.show()
