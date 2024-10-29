#Main Author @Atsushi Sakai 
# Implementation of Particle filter 

import sys 
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math 
import matplotlib.pyplot as plt
import numpy as np
from utility.angle import rot_mat_2d
from utility.plot import covariance_ellipse


class Robot():
    
    def __init__(self):
        self.model_covariance=np.diag([0.2])**2 #Range error
        self.measurement_covariance=np.diag([2.0,np.deg2rad(40.0)])**2 #input error


        #sim param
        self.Q_sim=np.diag([0.2])**2
        self.R_sim=np.diag([1.0,np.deg2rad(30.0)])**2

        self.dt=0.1 #time tick in s
        self.SIM_TIME=50.0
        self.MAX_RANGE=20.0

        #Particle filter params

        self.nP=100
        self.nTh=self.nP/2.0 #particles for re-sampling 


        self.show_anime=True 
        
class ParticleFilter(Robot):
    
    def control_input(self):
        v=1.0 
        yaw=0.1
        u=np.array([[v,yaw]]).T 
        return u
    
    def measurement(self,x_true,xd,u,landmarks):
    
        """
        x_true: the robot's current true state (position and orientation)
        xd: the robot dead-reckoned state 
        u: control input 
        landmarks: array of known landmarks, representing [x,y].
        
        z: The robot's measurement
        """
        
        x_true=self.motion_model(x_true,u)
        z=np.zeros((0,3))
        
        print(z)
        print(landmarks)
        
        for i in range(0,len(landmarks[:,0])):
            
            dx=x_true[0,0]-landmarks[i,0]
            dy=x_true[1,0]-landmarks[i,1]
            d= math.hypot(dx,dy) #Euclidean distance 
            
            if d<=self.MAX_RANGE: #if the landmark is within range
                
                noise=d + np.random.randn() * self.Q_sim[0,0] ** 0.5 #adding noise
                measurement_vector=np.array([[noise,landmarks[i,0],landmarks[i,1]]])
                z=np.vstack((z,measurement_vector)) #Stacking vector in our measurement array
                
        #add noise to input 
        ux= u[0,0]+np.random.randn() * self.R_sim[0,0] ** 0.5  #Adding Noise to the control input 
        uy= u[1,0]+np.random.randn() * self.R_sim[1,1] ** 0.5
        
        motion_vector= np.array([[ux,uy]]).T
        xd=self.motion_model(xd,motion_vector)
        
        return x_true,z,xd,motion_vector


    def motion_model(self,x,u):
        F = np.array([[1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, 0],
                    [0, 0, 0, 1]]) 
        B= np.array([[self.dt*math.cos(x[2,0]), 0],
                    [self.dt*math.sin(x[2,0]), 0],
                    [0, self.dt],
                    [0, 0]])
        
        
        x=F.dot(x)+B.dot(u)
        return x
        
    def gauss(self,x,sigma):
        prob = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
            math.exp(-x ** 2 / (2 * sigma ** 2))

        return prob 


    def calculate_covariance(self,x_est,px,pw):
        
        """
        x_est= Estimated mean 
        
        px= 2D array where each column represents different partice's state
        
        pw= 2D array where each column represents weights for different particles
        
        """
        
        cov=np.zeros((4,4))
        n_particles=px.shape[1]
        
        #Calculating covariance
        for i in range(n_particles):
            dx=(px[:,i:i+1]-x_est) #Select i th particle as column vector
            cov+= pw[0,i]*dx @ dx.T #@ refers to dot product
        cov*=1.0/(1.0-pw @ pw.T)
        
        return cov 


    def localization(self,px,pw,z,u):
        """
        LOCALIZATION WITH PARTICLE FILTER
        """ 
        
        for ip in range(self.nP):
            x= np.array([px[:,ip]]).T
            w=pw[0,ip]
            
            #PREDICTION STEP
            ux= u[0,0]+np.random.randn() * self.R_sim[0,0] ** 0.5  #Adding Noise to the control input 
            uy= u[1,0]+np.random.randn() * self.R_sim[1,1] ** 0.5      
            motion_vector= np.array([[ux,uy]]).T
            x=self.motion_model(x,motion_vector)
            
            #Calculate importance vector
            for i in range(len(z[:,0])):
                dx=x[0,0]-z[i,1]
                dy=x[1,0]-z[i,2]
                
                predict_z=math.hypot(dx,dy)
                
                dz=predict_z-z[i,0]
                w=w*self.gauss(dz,math.sqrt(self.model_covariance[0,0]))
                
            px[:,ip]=x[:,0]
            pw[0,ip]=w
            
            
        # pw=pw/pw.sum()
        weight_sum = pw.sum()
        if weight_sum == 0:
            print("Warning: Sum of weights is zero, resetting weights.")
            pw = np.ones_like(pw) / self.nP  
        else:
            pw = pw / weight_sum
            
        x_est=px.dot(pw.T)
            
        p_est=self.calculate_covariance(x_est,px,pw)
            
        N_eff=1.0/(pw.dot(pw.T))[0,0]
        if N_eff< self.nTh:
            px,pw= self.resampling(px,pw)
                
        return x_est,p_est,px,pw 

    def resampling(self,px,pw):
            #Low variance resampling 
            # from cyrill stachnis course
            
            cumulative_weight=np.cumsum(pw) #Cumsum computes cumulative weights
            base=np.arange(0.0,1.0,1/self.nP) #resampling is evenly between 0 to 1
            
            #Adding Random offset for resampling
            # This helps preventing clustering of particles
            resample_id= base + np.random.uniform(0,1/self.nP)
            
            #resampling indices:
            
            indexes=[]
            index=0
            for ip in range(self.nP): # Can be read as for ith particle in the entire particle array 
                while resample_id[ip]>cumulative_weight[index]:
                    index+=1
                    
                indexes.append(index)
            
            """
            new particle set will consist of particles that are more likely to 
            represent the current state, according to their weights.
            """ 
                
            px=px[:,indexes]
            
            """After resampling, the weights of the new particles are initialized 
            to be uniform. Each particle gets an equal weight.
            This is because, after resampling, all particles are considered equally important again.
            """
            
            pw=np.ones((1,self.nP))+1.0/self.nP
            
            
            return px,pw
        
    
    def plot_covariance_ellipse(self,x_est, p_est): 
        p_xy = p_est[0:2, 0:2]
        eig_val, eig_vec = np.linalg.eig(p_xy)

        if eig_val[0] >= eig_val[1]:
            big_ind = 0
            small_ind = 1
        else:
            big_ind = 1
            small_ind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        try:
            a = math.sqrt(eig_val[big_ind])
        except ValueError:
            a = 0

        try:
            b = math.sqrt(eig_val[small_ind])
        except ValueError:
            b = 0

        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
        fx = rot_mat_2d(angle) @ np.array([[x, y]])
        px = np.array(fx[:, 0] + x_est[0, 0]).flatten()
        py = np.array(fx[:, 1] + x_est[1, 0]).flatten()
        plt.plot(px, py, "--r")



def main():
    
    robot=ParticleFilter()
   
    time = 0.0
    landmark = np.array([[10.0, 0.0],
                         [10.0, 10.0],
                         [0.0, 15.0],
                         [-5.0, 20.0]])
    
    x_est = np.zeros((4, 1))
    x_true = np.zeros((4, 1))
    
    px = np.zeros((4, robot.nP))  # Particles store
    pw = np.zeros((1, robot.nP))  # Particles weight
    
    dead_reckon = np.zeros((4, 1))  # Dead reckoning

    # Storing history
    h_x_est = x_est
    h_x_true = x_true
    h_dead_reckon = dead_reckon
    
    while robot.SIM_TIME >= time:
        
        time += robot.dt
        u = robot.control_input()
        
        x_true, z, dead_reckon, motion_vector = robot.measurement(x_true, dead_reckon, u, landmark)
        x_est, p_est, px, pw = robot.localization(px, pw, z, motion_vector)
        
        # Store now
        h_x_est = np.hstack((h_x_est, x_est))
        h_dead_reckon = np.hstack((h_dead_reckon, dead_reckon))
        h_x_true = np.hstack((h_x_true, x_true))
        
        if robot.show_anime:
            plt.cla()
            
            # Stop with escape key
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            # Plot measurements
            for i in range(len(z[:, 0])):
                plt.plot([x_true[0, 0], z[i, 1]], [x_true[1, 0], z[i, 2]], "-k", label="Measurements" if i == 0 else "")
            
            plt.plot(landmark[:, 0], landmark[:, 1], "*k", label="Landmarks")
            plt.plot(px[0, :], px[1, :], ".r", label="Particles")
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "-b", label="True Trajectory")
            plt.plot(np.array(h_dead_reckon[0, :]).flatten(),
                     np.array(h_dead_reckon[1, :]).flatten(), "-k", label="Dead Reckoning Trajectory")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r", label="Estimated Trajectory (PF)")
            robot.plot_covariance_ellipse(x_est, p_est)
            
            plt.axis("equal")
            plt.grid(True)
            plt.legend(loc='upper right')  # Customize the location of the legend
            plt.title('Sensor Fusion Localization with Particle Filter')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.pause(0.001)

            
            
            
if __name__=="__main__":
    main()     
        
        
        

            
    


