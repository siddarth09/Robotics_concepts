# Kalman Filter

## Overview
The **Kalman Filter** is an algorithm that uses a series of measurements observed over time, containing noise (random variations), to produce estimates of unknown variables that tend to be more precise than those based on a single measurement alone. It is widely used in control systems, navigation, and robotics to estimate the state of a linear dynamic system.

## Key Concepts

### State Vector
The **state vector** represents the quantities we want to estimate, typically including position, velocity, and acceleration. For example, in a 2D scenario, the state vector can be represented as:

$$
\mathbf{x} = \begin{bmatrix}
p_x \\
p_y \\
v_x \\
v_y
\end{bmatrix}
$$

where:
- $p_x$ and $p_y$ are the positions in the x and y directions, respectively.
- $v_x$ and $v_y$ are the velocities in the x and y directions, respectively.

### Process Model
The **process model** describes how the state evolves over time. It can be mathematically expressed as:

$$\mathbf{x}_{k} = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{B} \mathbf{u}_{k} + \mathbf{w}_{k}$$


where:
- $\mathbf{F}$ is the state transition matrix that models the relationship between the previous state and the current state.
- $\mathbf{B}$ is the control input matrix.
- $\mathbf{u}_{k}$ represents the control inputs (like acceleration).
- $\mathbf{w}_{k}$ is the process noise, assumed to be Gaussian.

### Measurement Model
The **measurement model** relates the true state to the measurements obtained from sensors. It can be expressed as:

$$
\mathbf{z}_{k} = \mathbf{H} \mathbf{x}_{k} + \mathbf{v}_{k}
$$

where:
- $\mathbf{z}_{k}$ is the measurement vector.
- $\mathbf{H}$ is the observation matrix that maps the true state space into the observed space.
- $\mathbf{v}_{k}$ is the measurement noise, also assumed to be Gaussian.

### Kalman Filter Equations
The Kalman Filter operates in two main steps: **Prediction** and **Update**.

1. **Prediction Step**:
   - State Prediction:
   $$
   \mathbf{\hat{x}}_{k|k-1} = \mathbf{F} \mathbf{\hat{x}}_{k-1|k-1} + \mathbf{B} \mathbf{u}_{k}
   $$
   - Covariance Prediction:
   $$
   \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}
   $$

2. **Update Step**:
   - Kalman Gain:
   $$
   \mathbf{K}_{k} = \mathbf{P}_{k|k-1} \mathbf{H}^T \left( \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R} \right)^{-1}
   $$
   - State Update:
   $$
   \mathbf{\hat{x}}_{k|k} = \mathbf{\hat{x}}_{k|k-1} + \mathbf{K}_{k} \left( \mathbf{z}_{k} - \mathbf{H} \mathbf{\hat{x}}_{k|k-1} \right)
   $$
   - Covariance Update:
   $$
   \mathbf{P}_{k|k} = \left( \mathbf{I} - \mathbf{K}_{k} \mathbf{H} \right) \mathbf{P}_{k|k-1}
   $$

### Applications
The Kalman Filter is widely used in various fields, including:
- **Robotics**: For localization and mapping.
- **Navigation**: In GPS and inertial navigation systems.
- **Economics**: For estimating economic indicators over time.


### IMPLEMENTATION OF KALMAN FILTER ON GPS DATA (humdle-dev branch)
[RUSTY](https://github.com/siddarth09/Rusty)


![kalman_filter](https://github.com/user-attachments/assets/bb8027e4-cde2-4771-981b-8f45a44953f7)


### Conclusion
The Kalman Filter provides a powerful method for estimating the state of a system in the presence of noise. Its recursive nature makes it suitable for real-time applications, allowing it to be implemented in various technologies that require state estimation and prediction.


