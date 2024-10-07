import numpy as np
import matplotlib.pyplot as plt

"""
Bayes filter is a probabilistic method used to estimate the state of a system over time 
using noisy sensor data and control inputs. It alternates between two steps:

1. Prediction: Updates the belief about the system's state based on the control input 
   and a motion model, which increases uncertainty due to possible errors in motion.
   
2. Update: Refines the predicted belief using the latest sensor measurements, reducing 
   uncertainty by incorporating real-world data.

This recursive process allows continuous state estimation in dynamic systems with noise.
"""

class BayesFilter():

    def __init__(self, state_space, control_motion, motion_noise, sensor_measurement, measurement_noise):
        self.state_space = state_space
        self.control_motion = control_motion
        self.motion_noise = motion_noise
        self.sensor_measurement = sensor_measurement
        self.measurement_noise = measurement_noise

    def motion_model(self, state, control, motion_noise):
        return state + control + np.random.normal(0, motion_noise)
    
    def measurement_model(self, actual_z, measurement_noise):
        return actual_z + np.random.normal(0, measurement_noise)
    
    def gaussian(self, x, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def bayes_filter(self, prior_belief):
        # Prediction step
        predicted_belief = np.zeros_like(self.state_space)
        for i, state in enumerate(self.state_space):
            for j, previous_state in enumerate(self.state_space):
                predicted_state = self.motion_model(previous_state, self.control_motion, self.motion_noise)
                predicted_belief[i] += prior_belief[j] * self.gaussian(predicted_state, state, self.motion_noise)

        # Update step
        updated_belief = np.zeros_like(predicted_belief)
        for i, state in enumerate(self.state_space):
            likelihood = self.gaussian(self.sensor_measurement, state, self.measurement_noise)
            updated_belief[i] = predicted_belief[i] * likelihood

        # Normalize belief
        updated_belief /= np.sum(updated_belief)

        return updated_belief


def animate_bayes_filter_with_trajectory():
    # Define the state space (e.g., position)
    state_space = np.linspace(1.0, 10.0, 100)
    
    # Prior belief (uniform distribution)
    prior_belief = np.ones_like(state_space) / len(state_space)
    
    # Control parameters
    control_motion = 0.5  # control input (movement per time step)
    motion_noise = 0.3  # motion model noise

    # Actual measurement and sensor noise
    actual_z = 1.0  # initial actual position of the robot
    measurement_noise = 0.1  # measurement noise

    # Initialize the Bayes filter
    bayes_filter = BayesFilter(state_space, control_motion, motion_noise, actual_z, measurement_noise)

    # Set up the plot for animation
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot for the belief distribution
    line_belief, = ax1.plot(state_space, prior_belief, label="Belief")
    ax1.set_ylim(0, 0.5)
    ax1.set_xlim(state_space[0], state_space[-1])
    ax1.set_xlabel('State (Position)')
    ax1.set_ylabel('Belief Probability')
    ax1.set_title('Bayes Filter Belief Evolution')
    ax1.grid(True)

    # Plot for the robot's trajectory
    trajectory = [actual_z]
    trajectory_line, = ax2.plot(trajectory, np.zeros_like(trajectory), 'ro-', label='Robot Position')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('Position')
    ax2.set_title('Robot Trajectory')
    ax2.grid(True)

    # Animation loop
    for t in range(50):
        # Simulate new actual position and measurement
        actual_z += control_motion
        measurement_z = bayes_filter.measurement_model(actual_z, measurement_noise)

        # Update the belief using the Bayes filter
        bayes_filter.sensor_measurement = measurement_z
        new_belief = bayes_filter.bayes_filter(prior_belief)

        # Update belief plot
        line_belief.set_ydata(new_belief)
        prior_belief = new_belief

        # Update the trajectory plot
        trajectory.append(actual_z)
        trajectory_line.set_xdata(trajectory)
        trajectory_line.set_ydata(np.zeros_like(trajectory))

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(0.1)

    plt.ioff()
    plt.show()


def main():
    animate_bayes_filter_with_trajectory()


if __name__ == "__main__":
    main()
