import numpy as np 
import matplotlib.pyplot as plt 

class ParticleFilter():
    def __init__(self,state_dim,num_particles,r,w,sigma_l,sigma_r,measurement_noise):
        self.num_particles = num_particles #No. of particles
        self.r = r #The wheel radius
        self.w = w #The track width 
        self.sigma_l = sigma_l #The left wheel speed
        self.sigma_r = sigma_r #The right wheel speed
        self.measurement_noise = measurement_noise #Measurement Noise (R=with mean mu, and sigma)
        # self.particles = np.array([(np.random.uniform(-1, 1), 
        #                     np.random.uniform(-1, 1),
        #                     np.random.uniform(-np.pi, np.pi)) for _ in range(num_particles)])
        self.particles = np.array([(0,0,0) for _ in range(num_particles)])

        
        
        self.state_space=np.zeros((state_dim,1))
    def sample_speeds(self,vl,vr):
            
        """
        Wheel left and right speeds
        args:
        vl (float): Left wheel speed
        vr (float): Right wheel speed
            
        returns:
            (float,float): Noise added to speeds
        """
            
        vl_noise=vl+np.random.normal(0,self.sigma_l)
        vr_noise=vr+np.random.normal(0,self.sigma_r)
        
        return vl_noise, vr_noise
    
    def motion_model(self,particles,vl,vr,t1,t2):
        """
        Motion model for the particles
        Args:
            particles (tuple_): current particle state (x,y,theta)
            vl (float): left speed command
            vr (float): right speed command
            t1 (float): Current Time
            t2 (float): Future time 
            
        Return: 
         tuple: New particle state (x,y,theta)at time t2
        """
        x,y,theta=particles
        
        dt=t2-t1
        
        vl_noisy,vr_noisy=self.sample_speeds(vl,vr)
        
        v=0.5*self.r*(vl_noisy+vr_noisy) #linear velocity
        w=0.5 * self.r * (vr_noisy - vl_noisy)/self.w #angular velocity 
        
        
        if abs(w) < 1e-5:
            dx=v*dt*np.cos(theta)
            dy=v*dt*np.sin(theta)
            dtheta=0
        else:
            dx=(v/w)*(np.sin(theta+w*dt)-np.sin(theta))
            dy=(v/w)*(np.cos(theta)-np.cos(theta+w*dt))
            dtheta=w*dt 
            
            
        x_new=x+dx
        y_new=y+dy
        theta_new=theta+dtheta
        
        self.state_space=[x_new,y_new,theta_new]
        
        return (x_new,y_new,theta_new)   
    
    
    def predict(self,vel_l,vel_r,t1,t2):
        """
        Predicted all particles based on the motion model 
        
        args:  
        
            vel_l (float): left wheel speed
            vel_r (float): right wheel speed
            t1 (float): Current Time
            t2 (float): future time 
        """
        
        self.particles=[self.motion_model(particle,vel_l,vel_r,t1,t2) 
                        for particle in self.particles]
        
        
    def measurement_model(self, measurement, particle):
        """
        Measurement Model that calculates the likelihood of the observed measurement
        given the particle's state using a Gaussian model.
        
        Args:
            particle (tuple): Particle state (x, y, theta)
            measurement (np.ndarray): Observed measurement (x, y) with noise
            
        Returns:
            float: Likelihood of the measurement given the particle's state
        """
        
        
        
        x, y = particle[:2]        
       
        new_measurement = np.linalg.norm(np.array([x, y]) - measurement[:1])
        gaussian_weight = (1 / (2 * np.pi * self.measurement_noise**2)) * np.exp(-0.5 * (new_measurement**2) / self.measurement_noise**2)
            
        # print(f"Measured distance: {new_measurement}, Likelihood (weight): {gaussian_weight}")
        return gaussian_weight

    
    def resample_particles(self, weights):
        """Resample particles based on importance weights."""
        
        
        weights = np.array(weights)
        # weights /= np.sum(weights) 
        
        weights_sum= np.sum(weights)
        if weights_sum == 0:
            print("All weights are zero! Check measurement model.")
            weights = np.ones(self.num_particles) / self.num_particles  # Reset to uniform
        else:
            weights /= weights_sum 
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        resampled_particles = [self.particles[i] for i in indices]
        
        
        self.particles = resampled_particles

    def low_variance_resample(self, weights):
        """Low variance resampling method."""
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            print("All weights are zero! Check measurement model.")
            weights = np.ones(self.num_particles) / self.num_particles  
        else:
            weights /= weights_sum
        
        num_particles = self.num_particles
        resampled_particles = []
    
        r = np.random.uniform(0, 1/num_particles)  
        cumulative_sum = 0.0
        index = 0
        c = weights[0]
    
        for i in range(num_particles):
            U = r + i / num_particles 
            while U > c:
                index += 1
                c += weights[index]
            resampled_particles.append(self.particles[index])
    
        self.particles = resampled_particles

    
    def update(self, measurement):
        """
        Update step

        Args:
            measurement (np.array): Noisy Measurement of the robot's position 
        """
        
        weights = []

        
        for particle in self.particles:
            weight = self.measurement_model(measurement, particle)
            weights.append(weight)

        weights = np.array(weights)

        
        self.resample_particles(weights)
        
    def mu_covar(self):
        
        positions=np.array(self.particles)[:, :2] 
        
        mu=np.mean(positions, axis=0)
        cov=np.cov(positions.T)
        
        
        
        return mu, cov
    
    def e_plot_particles(self):
       
        positions = np.array(self.particles)[:, :2]
        
        
        plt.figure(figsize=(8, 6))
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=5, label='Particles')
        plt.title('Particle Positions at Time t = 10')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.grid(True)
        plt.legend()
        plt.show()


        
        

def main():
    
    num_particles = 1000
    state_dim=3
    r = 0.25 # Wheel radius
    w = 0.5   # Track width
    sigma_l = 0.05  #left wheel noise
    sigma_r = 0.05  # right wheel noise
    measurement_noise = 0.10   # Standard deviation for measurement noise

    
    pf = ParticleFilter(state_dim,num_particles,r,w,sigma_l,sigma_r,measurement_noise)

   
    v_l, v_r = 1.5, 2.0  # rad/s

    # Time interval
    t1 = 0.0  # Current time
    t2 = 10.0  # Future time

   
    pf.predict(v_l,v_r,t1,t2)
   
    z = np.array([0.5, 0.2])

    
    pf.update(z)

    # Get the updated particle set
    updated_particles = pf.particles

    # Print the updated particle set
    print("Updated particle set:")
    for i, particle in enumerate(updated_particles[:5]): 
        print(f"Particle {i + 1}: {particle}")
        
        
    mean, covariance = pf.mu_covar()
    
   
    print(f"Empirical Mean: {mean}")
    print(f"Empirical Covariance Matrix: \n{covariance}")
    
   
    pf.e_plot_particles()
        
if __name__=="__main__":
    main()
