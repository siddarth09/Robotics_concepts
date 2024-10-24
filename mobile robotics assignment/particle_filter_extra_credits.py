from particle_filter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt 

def plot_particles(means,all_positions, times):
    """Plot the particles for different times and overlay the trajectory."""
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(8, 6))
    
    # Plot the particle positions at each time step
    for i, (positions, time) in enumerate(zip(all_positions, times)):
        plt.scatter(positions[:, 0], positions[:, 1], c=colors[i], s=5, label=f't = {time}')
    
    # Plot the robot's trajectory based on empirical means
    means = np.array(means)
    plt.plot(means[:, 0], means[:, 1], 'k-', label='Robot trajectory')  # Black line for trajectory
    plt.scatter(means[:, 0], means[:, 1], c='black', label='Mean positions', zorder=5)  # Highlight mean positions
    
    plt.title('Particle Positions and Robot Trajectory Over Time')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.grid(True)
    plt.legend()
    plt.show()

def question_f():
    
    
    print("question f solution")
    num_particles = 1000
    state_dim = 3
    r = 0.25  # Wheel radius
    w = 0.5   # Track width
    sigma_l = 0.05  # Left wheel noise
    sigma_r = 0.05  # Right wheel noise
    measurement_noise = 0.10  # Standard deviation for measurement noise

    pf = ParticleFilter(state_dim, num_particles, r, w, sigma_l, sigma_r, measurement_noise)

    v_l, v_r = 1.5, 2.0  # Rad/s wheel speeds

    # Time intervals to predict (5, 10, 15, 20 seconds)
    times = [5, 10, 15, 20]
    all_positions = []
    means=[]

    # Predict the particle states at each time step and store the results
    for t in times:
        pf.predict(v_l, v_r, 0, t)  # Predict particle states
        positions = np.array(pf.particles)[:, :2]  # Extract (x, y) positions
        all_positions.append(positions)
        
        # Calculate and print empirical mean and covariance
        mean, covariance = pf.mu_covar()
        means.append(mean)
        print(f"Time t = {t} seconds")
        print(f"Empirical Mean: {mean}")
        print(f"Empirical Covariance Matrix: \n{covariance}\n")

    # Plot the particles at different times
    plot_particles(means,all_positions, times)


def question_g():
    print("question g solution")
    
    num_particles = 1000
    state_dim = 3
    r = 0.25  # Wheel radius
    w = 0.5   # Track width
    sigma_l = 0.05  # Left wheel noise
    sigma_r = 0.05  # Right wheel noise
    measurement_noise = 0.10  # Standard deviation for measurement noise

    pf = ParticleFilter(state_dim, num_particles, r, w, sigma_l, sigma_r, measurement_noise)

    v_l, v_r = 1.5, 2.0  # Rad/s wheel speeds

    
    times = [5, 10, 15, 20]
    measurements = [
        np.array([1.6561, 1.2847]),
        np.array([1.0505, 3.1059]),
        np.array([-0.9875, 3.2118]),
        np.array([-1.6450, 1.1978])
    ]
    
    all_positions = []
    mus = []
    covariances = []

   
    for t, z in zip(times, measurements):
        pf.predict(v_l, v_r, 0, t) 
        positions = np.array(pf.particles)[:, :2]  
        all_positions.append(positions)

        pf.update(z)  
        
        mu, covariance = pf.mu_covar()
        mus.append(mu)
        covariances.append(covariance)

        print(f"Time t = {t} seconds")
        print(f"Measurement z_t = {z}")
        print(f"Empirical Mean: {mu}")
        print(f"Empirical Covariance Matrix: \n{covariance}\n")
        
    plot_particles(mus,all_positions,times)
    
    
def main():
    question_f()
    
if __name__ == '__main__':
    main()