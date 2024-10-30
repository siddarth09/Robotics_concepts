
# PARTICLE FILTER

A **Particle Filter** is an algorithm used to estimate the state of a system (such as a robot's position) over time, particularly when measurements are noisy, and there’s uncertainty in movement. It represents the state as a collection of particles, each with a possible pose. Here's how it works:

1. **Initialization**:
   - Start with a set of particles \( N \) representing the robot's initial belief of its position. For example, if the robot starts at the origin with certainty, all particles are initialized to the origin.
   - Each particle has a state vector \((x, y, \theta)\), representing the \(x\) and \(y\) position and orientation \(\theta\) of the robot.

2. **Prediction**:
   - At each time step, predict the robot's new position based on the control inputs (e.g., wheel velocities) and a motion model. 
   - For each particle, calculate its next position by applying the motion model, adding noise to simulate real-world uncertainty in movement. This noise can represent the effect of factors like wheel slippage.
   - For a differential-drive robot, the updated position \((x', y', \theta')\) of each particle is computed using the current velocities and wheel radius, track width, and angular displacement.

3. **Measurement Update (Importance Weighting)**:
   - Once a measurement (e.g., GPS or rangefinder) is received, update each particle’s weight based on its likelihood given this new measurement.
   - Use a measurement model (such as Gaussian distribution) to compute the probability that each particle's predicted measurement matches the actual observed measurement. This gives each particle a weight indicating its likelihood of representing the robot's true position.

4. **Resampling**:
   - Resample particles based on their weights, favoring those with higher weights, which are more likely to represent the robot’s actual position. Particles with low weights are less likely to be sampled and may be discarded.
   - Common resampling techniques include:
     - **Multinomial Resampling** (standard): Particles are resampled proportionally to their weights.
     - **Low-Variance Resampling** (common in robotics): This approach reduces variance by systematically selecting particles, avoiding large concentrations of particles in low-probability areas.

5. **Estimate the Robot’s Pose**:
   - After resampling, compute the **mean** and **covariance** of the particle positions. The mean serves as an estimate of the robot's position, while the covariance gives a measure of the uncertainty in this estimate.
   - Repeat the prediction, update, and resampling steps at each time step as the robot moves and collects more measurements.

6. **Visualization and Results**:
   - The final distribution of particles represents the posterior belief of the robot's position. The spread (covariance) of particles shows the level of uncertainty.
   - For visualization, plotting the trajectory (mean positions) over time can show the robot’s estimated path.
-----

## Visualization:

![Screencast from 10-29-2024 09_23_06 PM](https://github.com/user-attachments/assets/41946073-a46c-4b85-b39b-daecb9d737a5)

