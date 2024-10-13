# Bayes Filter

A **Bayes filter** is a probabilistic framework used to estimate the state of a dynamic system over time, based on noisy sensor measurements and control inputs. It operates by recursively updating a belief (a probability distribution) of the system's state, considering both the uncertainty in motion (from control inputs) and the uncertainty in measurements (from sensors).

At the core of a Bayes filter is the concept of belief `bel(x_t)`, which represents the probability distribution over the state `x_t` at time `t`, given all prior control inputs and observations. The Bayes filter alternates between two phases:

### 1. Prediction Step:

The system's state is predicted based on a motion model and the control input `u_t`. This step updates the prior belief to account for the expected motion of the system but increases uncertainty due to potential errors in control.

![prediction step](https://github.com/user-attachments/assets/3e29fd3d-46a2-44b3-8857-b02438ee9dd4)
Here, `p(x_t \mid u_t, x_{t-1})` is the probability of transitioning from `x_{t-1}` to `x_t` given the control input `u_t`.


![bayes_fil1](https://github.com/user-attachments/assets/80aca5c6-e55d-4f6c-a3f8-f475763a2ff9)

### 2. Update Step:

In this step, the predicted belief is updated using the actual sensor measurement `z_t` to correct the estimate. This reduces uncertainty by incorporating real-world observations.


![update step](https://github.com/user-attachments/assets/1552fd16-0d52-450d-8b58-7f4bebc857a5)

Here, `p(z_t \mid x_t)` is the likelihood of observing `z_t` given the state `x_t`, and `\eta` is a normalization factor ensuring the belief sums to 1.


![bayes_fil2](https://github.com/user-attachments/assets/035915f6-ce4c-4a8e-b099-b2f19780884b)

### Summary

By repeating these steps over time, the Bayes filter provides a way to continuously estimate the system's state in the presence of uncertainty. This method is fundamental for many applications in robotics and autonomous systems.


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

### Conclusion
The Kalman Filter provides a powerful method for estimating the state of a system in the presence of noise. Its recursive nature makes it suitable for real-time applications, allowing it to be implemented in various technologies that require state estimation and prediction.





