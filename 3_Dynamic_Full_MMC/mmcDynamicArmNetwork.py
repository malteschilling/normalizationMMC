"""
A dynamic MMC network for a three segmented manipulator.

Using a vector notation.
For normalization: train a simple feedforward NN for normalization and incorporate this
into the overall structure of the network.
Furthermore, incorporates velocity equation which lead to biological movement characteristics.

Malte Schilling, 5.1.2019

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import keras
from keras.models import load_model

class mmcDynamicArmNetwork:

    ''' Initialisation of the body model.
    
        Can be called from the outside with the name for the normalization layer NN h5 file.
        If none is given, normalization from linalg is used.
        
        The model is for a three segmented arm with each segment of unit length.
        
        Model follows the MMC approach:
            - multiple computations (redundant)
            - are integrated through mean calculation
            - and afterwards a normalization is required
    '''
    def __init__(self, norm_model_name=None):
        self.damping = 10.
        self.vel_damping = 5.
        # To avoid overshoot: we introduced a consistent velocity decay term. 
        # It always decreases the velocity by a constant fraction through multiplication 
        # with a decay factor (set to 0.92). 
        # This acts like a constant friction as the velocity of all joints is slightly decreased all the time.
        self.vel_decay_direct = 0.92
        self.current_timestep = 0

        self.reset_all_variables()
        # Turning on matplotlib
        self.live_plot = False
        # Model with 16 hidden neurons: loss: 0.0026 - val_loss: 0.0030
        # Model with 8 hidden neurons: loss: 0.0035 - val_loss: 0.0054
        if norm_model_name!=None:
            self.normalization_model = load_model('../1_Normalization_MLPs/trained_normalization_subnetworks/' + norm_model_name)
            print("Loaded model ", norm_model_name)
        else:
            self.normalization_model = None
    
    def reset_all_variables(self):
        # Initial configuration of the arm
        self.l_1 = np.array([0.99,0.15])
        self.l_2 = np.array([0.707,0.7071])
        self.l_3 = np.array([0.15,0.99])
        self.l_1_norm = np.array(self.l_1)
        self.l_2_norm = np.array(self.l_2)
        self.l_3_norm = np.array(self.l_3)
        self.l_1_hat = np.array(self.l_1)
        self.l_2_hat = np.array(self.l_2)
        self.l_3_hat = np.array(self.l_3)
        self.r = np.array( self.l_1 + self.l_2 + self.l_3 )
        self.r_old = np.array(self.r)
        self.target_r = np.array(self.r)
        self.d_1 = np.array( self.l_1 + self.l_2)
        self.d_2 = np.array( self.l_2 + self.l_3 )
        
        self.vel_l_1 = np.array([0.,0.])
        self.vel_l_2 = np.array([0.,0.])
        self.vel_l_3 = np.array([0.,0.])
        self.vel_l_1_hat = np.array([0.,0.])
        self.vel_l_2_hat = np.array([0.,0.])
        self.vel_l_3_hat = np.array([0.,0.])
        
        self.reset_multiple_calculations()
        
    def reset_multiple_calculations(self):
        # The results of the multiple computations are appended into lists
        self.l_1_calc = []
        self.l_2_calc = []
        self.l_3_calc = []
        self.d_1_calc = []
        self.d_2_calc = []
        self.r_calc = []
  
    """ The Main MMC step:
            - multiple computations are down in compute_vector_equation_step
            - normalization is done in between! - because it uses as an input the old value
            - afterwards integration is done in weighted_update_variables
    """      
    def mmc_iteration_step(self, timesteps=100):
        norm_dist = np.linalg.norm(self.r - self.target_r)
        self.r = np.array(self.target_r)
        self.current_timestep = 0
        # Track movement and velocity = normalized over the maximum distance for that movement.
        self.velocity_r = np.zeros(timesteps)
        self.distance_target = np.zeros(timesteps)
        for i in range(0, timesteps):
            if self.live_plot:
                plt.pause(0.1)
                self.draw_manipulator()
            self.reset_multiple_calculations()
            # Multiple computation step
            self.compute_vector_equation_step()
            # Normalization step (acts on current values - delayed output)
            if self.normalization_model:
                self.compute_normalization_step()
            else:
                self.compute_linalg_normalization_step()
            # Integration step: Mean calculation
            self.weighted_update_variables()
            # Introduced velocity step:
            # Calculate new velocities which are then integrated into the update step
            self.velocity_step()
            
            self.update_variables()
        return (self.distance_target/norm_dist, self.velocity_r/norm_dist)
    
    """ Multiple computations
         The different local triangle equations for the arm.
    """    
    def compute_vector_equation_step(self):
        self.l_1_calc.append( self.d_1 - self.l_2_norm )
        self.l_1_calc.append( self.r - self.d_2 )
        
        self.l_2_calc.append( self.d_1 - self.l_1_norm )
        self.l_2_calc.append( self.d_2 - self.l_3_norm )
        
        self.l_3_calc.append( self.d_2 - self.l_2_norm )
        self.l_3_calc.append( self.r - self.d_1 )
        
        self.d_1_calc.append( self.l_1_norm + self.l_2_norm )
        self.d_1_calc.append( self.r - self.l_3_norm )
        
        self.d_2_calc.append( self.r - self.l_1_norm )
        self.d_2_calc.append( self.l_2_norm + self.l_3_norm )
        
        self.r_calc.append( self.l_1_norm + self.d_2 )
        self.r_calc.append( self.d_1 + self.l_3_norm )
        #print(self.l_1_calc, self.r, self.d_1, self.d_2)
        
    """ Normalization using a simple ffw neural network (loaded from file).
    """
    def compute_normalization_step(self):
        #print("Prediction: ", self.normalization_model.predict(Xnew), " - ", self.l_1, (self.l_1/np.linalg.norm(self.l_1)))
        self.l_1_norm = np.squeeze(self.normalization_model.predict(np.expand_dims(self.l_1_hat, axis=0))) #self.l_1/np.linalg.norm(self.l_1)
        self.l_2_norm = np.squeeze(self.normalization_model.predict(np.expand_dims(self.l_2_hat, axis=0))) #self.l_2/np.linalg.norm(self.l_2)
        self.l_3_norm = np.squeeze(self.normalization_model.predict(np.expand_dims(self.l_3_hat, axis=0))) #self.l_3/np.linalg.norm(self.l_3)
        #print(self.l_1_norm - self.l_1/np.linalg.norm(self.l_1))
    
    """ Normalization """
    def compute_linalg_normalization_step(self):
        self.l_1_norm = self.l_1_hat/np.linalg.norm(self.l_1_hat)
        self.l_2_norm = self.l_2_hat/np.linalg.norm(self.l_2_hat)
        self.l_3_norm = self.l_3_hat/np.linalg.norm(self.l_3_hat)
        #print(self.l_1_norm - self.l_1/np.linalg.norm(self.l_1))
    
    """ Mean calculation:
         Integration of different equations as well as recurrency.
    """
    def weighted_update_variables(self):
        self.l_1 = np.sum(self.l_1_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.l_1_norm
        self.l_2 = np.sum(self.l_2_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.l_2_norm
        self.l_3 = np.sum(self.l_3_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.l_3_norm
        
        self.d_1 = np.sum(self.d_1_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.d_1
        self.d_2 = np.sum(self.d_2_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.d_2
        # Turned of for inverse kinematic case
        #self.r = np.sum(self.r_calc, axis=0)/self.damping + ((self.damping-2.)/self.damping) * self.r

    """ Calculation of velocity etc.
    """
    def velocity_step(self):
        #print((self.l_1 - self.l_1_norm), self.vel_l_1)
        self.vel_l_1 = self.vel_decay_direct *(((self.l_1 - self.l_1_norm)/(self.vel_damping + 1)) + self.vel_l_1*(self.vel_damping/(self.vel_damping+1)))
        #print(self.vel_l_1, ((self.l_1 - self.l_1_norm)/(self.vel_damping + 1)), self.vel_l_1, (self.vel_damping/(self.vel_damping+1)), (self.vel_l_1*(self.vel_damping/(self.vel_damping+1))) )
        self.vel_l_2 = self.vel_decay_direct * ((self.l_2 - self.l_2_norm)/(self.vel_damping + 1) + self.vel_l_2*(self.vel_damping/(self.vel_damping+1)))
        self.vel_l_3 = self.vel_decay_direct * ((self.l_3 - self.l_3_norm)/(self.vel_damping + 1) + self.vel_l_3*(self.vel_damping/(self.vel_damping+1)))
        self.l_1_hat = self.l_1_norm + self.vel_l_1
        self.l_2_hat = self.l_2_norm + self.vel_l_2
        self.l_3_hat = self.l_3_norm + self.vel_l_3
    
    """ Calculation of velocity etc.
    """
    def update_variables(self):
        self.velocity_r[self.current_timestep] = np.abs(np.linalg.norm( (self.l_1_norm + self.l_2_norm + self.l_3_norm) - self.r_old ) )
        self.distance_target[self.current_timestep] = np.abs(np.linalg.norm( self.target_r - (self.l_1_norm + self.l_2_norm + self.l_3_norm) ))
        self.r_old = (self.l_1_norm + self.l_2_norm + self.l_3_norm)
        
        self.current_timestep += 1
 
    """ **** Graphic methods: Simple drawing of the body model **************************
    """       
    def plot_velocity_distance(self):
        fig = plt.figure(figsize=(8, 6))
        ax_dist = plt.subplot(111)  
        #ax_arch.spines["top"].set_visible(False)  
        #ax_arch.spines["right"].set_visible(False)  
        #ax_arch.set_xticks(np.arange(0,len(hist_list)))
        plt.plot(range(0, len(self.distance_target)), self.distance_target, color=(1., 187./255, 120./255), lw=2)
        ax_dist.set_xlabel('Iteration steps', fontsize=14)
        ax_dist.set_ylabel('Distance to target', fontsize=14)
        fig2 = plt.figure(figsize=(8, 6))
        ax_vel = plt.subplot(111)  
        plt.plot(range(0, len(self.velocity_r)), self.velocity_r, color=(31/255, 119./255, 180./255), lw=2)
        ax_vel.set_xlabel('Iteration steps', fontsize=14)
        ax_vel.set_ylabel('Velocity end effector', fontsize=14)
        #ax_arch.set_title('MSE after Learning', fontsize=20)   
        plt.show()
        
    """ Initialising the drawing window.
            Must be called before the visualisation can be updated
            by calling draw_manipulator"""			
    def initialise_drawing_window(self):
        fig = plt.figure(figsize=(6, 6))
        self.plot_target_r, = plt.plot([0, self.target_r[0]], [0, self.target_r[1]], linestyle=':', linewidth=1.0, color='gray', marker='x')
        self.plot_arm, = plt.plot([0,0], [0,0], linestyle='-', linewidth=12.0, color='gray', alpha=0.75, solid_capstyle='round', marker='o')
        plt.xlim(-2.,2.)  
        plt.ylim(-0.5,3.5)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().get_xaxis().set_visible(False)
        self.live_plot = True

    """ The draw method for the manipulator arm.
            It is called from the outside iteration loop. """		
    def draw_manipulator(self):
        # Draw permanently current arm configuration
        if (self.current_timestep % 2 == 1):
            plt.plot([0, self.l_1_norm[0], self.l_1_norm[0]+self.l_2_norm[0], self.l_1_norm[0]+self.l_2_norm[0]+self.l_3_norm[0]],
                [0, self.l_1_norm[1], self.l_1_norm[1]+self.l_2_norm[1], self.l_1_norm[1]+self.l_2_norm[1]+self.l_3_norm[1]], 
                linestyle='--', linewidth=1.0, color='gray')
        
        self.plot_arm.set_xdata([0, self.l_1_norm[0], self.l_1_norm[0]+self.l_2_norm[0], self.l_1_norm[0]+self.l_2_norm[0]+self.l_3_norm[0]])
        self.plot_arm.set_ydata([0, self.l_1_norm[1], self.l_1_norm[1]+self.l_2_norm[1], self.l_1_norm[1]+self.l_2_norm[1]+self.l_3_norm[1]])
        #plt.savefig("/Users/mschilling/Desktop/Dynamic_Example_Movement.pdf")
    
    def set_new_target(self, new_target):
        self.target_r = np.array(new_target)
        plt.plot([0, self.target_r[0]], [0, self.target_r[1]], linestyle=':', linewidth=1.0, color='gray', marker='x')

    


        