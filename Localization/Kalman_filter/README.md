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




# Extended Kalman Filter (EKF)

## Overview
The **Extended Kalman Filter (EKF)** is a nonlinear version of the Kalman Filter, designed to handle systems where the process or measurement models are nonlinear. While the Kalman Filter is used for linear systems, the EKF extends its applicability to nonlinear systems by linearizing the process and measurement models around the current estimate.

EKF is commonly used in robotics, navigation, and tracking systems where nonlinearity is a factor, such as when dealing with sensor data like GPS or IMU.

## Key Concepts

### State Vector
Like the standard Kalman Filter, the **state vector** in EKF represents the quantities we aim to estimate. For example, in a 2D scenario, the state vector can represent the position and velocity of a robot:

$$
\mathbf{x} = \begin{bmatrix}
p_x \\
p_y \\
v_x \\
v_y
\end{bmatrix}
$$

where:
- \( p_x \) and \( p_y \) are the positions in the x and y directions.
- \( v_x \) and \( v_y \) are the velocities in the x and y directions.

### Process Model (Nonlinear)
The **process model** in EKF describes how the state evolves over time, but since the system is nonlinear, it can be represented by a nonlinear function:

$$
\mathbf{x}_{k} = g(\mathbf{x}_{k-1}, \mathbf{u}_{k}) + \mathbf{w}_{k}
$$

where:
- \( g(\mathbf{x}_{k-1}, \mathbf{u}_{k}) \) is the nonlinear state transition function.
- \( \mathbf{u}_{k} \) is the control input (such as acceleration).
- \( \mathbf{w}_{k} \) is the process noise, assumed to be Gaussian.

### Measurement Model (Nonlinear)
The **measurement model** also becomes nonlinear in EKF, describing how the sensor observations relate to the true state:

$$
\mathbf{z}_{k} = h(\mathbf{x}_{k}) + \mathbf{v}_{k}
$$

where:
- \( h(\mathbf{x}_{k}) \) is the nonlinear measurement function.
- \( \mathbf{v}_{k} \) is the measurement noise, assumed to be Gaussian.

### Jacobians
Since the models are nonlinear, EKF uses **Jacobians** to linearize the process and measurement models around the current state estimate.

- **Jacobian of the process model**: 
$$ 
\mathbf{G} = \frac{\partial g}{\partial \mathbf{x}} 
$$
- **Jacobian of the measurement model**: 
$$ 
\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}} 
$$

These matrices approximate the local linearity of the nonlinear functions and are used in the prediction and update steps.

### Extended Kalman Filter Equations
Similar to the standard Kalman Filter, EKF operates in two main steps: **Prediction** and **Update**. but instead of using a linear model, it linearizes using jacobians.

1. **Prediction Step**:
   - State Prediction:
   $$
   \mathbf{\hat{x}}_{k|k-1} = g(\mathbf{\hat{x}}_{k-1|k-1}, \mathbf{u}_{k})
   $$
   - Covariance Prediction:
   $$
   \mathbf{P}_{k|k-1} = \mathbf{G}_{k} \mathbf{P}_{k-1|k-1} \mathbf{G}_{k}^T + \mathbf{Q}
   $$
   where \( \mathbf{G}_{k} \) is the Jacobian of the process model.

2. **Update Step**:
   - Kalman Gain:
   $$
   \mathbf{K}_{k} = \mathbf{P}_{k|k-1} \mathbf{H}_{k}^T \left( \mathbf{H}_{k} \mathbf{P}_{k|k-1} \mathbf{H}_{k}^T + \mathbf{R} \right)^{-1}
   $$
   - State Update:
   $$
   \mathbf{\hat{x}}_{k|k} = \mathbf{\hat{x}}_{k|k-1} + \mathbf{K}_{k} \left( \mathbf{z}_{k} - h(\mathbf{\hat{x}}_{k|k-1}) \right)
   $$
   - Covariance Update:
   $$
   \mathbf{P}_{k|k} = \left( \mathbf{I} - \mathbf{K}_{k} \mathbf{H}_{k} \right) \mathbf{P}_{k|k-1}
   $$

### Applications
The EKF is widely used in nonlinear systems for:
- **Mobile Robotics**: For localization and tracking where sensors like lidar and GPS have nonlinear behavior.
- **Navigation**: In systems like GPS/IMU fusion, where measurements and motion models are nonlinear.
- **Sensor Fusion**: Combining multiple noisy sensor readings for accurate state estimation.

### Example: EKF for Localization
In a typical localization scenario, the state vector \( \mathbf{x} \) could represent the position and velocity of a robot, while GPS data provides noisy measurements of position. The EKF would predict the robot's position based on its current velocity and control inputs (acceleration), and then update the estimate using the GPS measurements.

#### Process Model (Motion Model):
For a robot moving in 2D, the process model might look like:
$$
\mathbf{x}_{k} = g(\mathbf{x}_{k-1}, \mathbf{u}_{k}) = \begin{bmatrix}
p_x + v_x \cdot \Delta t \\
p_y + v_y \cdot \Delta t \\
v_x + a_x \cdot \Delta t \\
v_y + a_y \cdot \Delta t
\end{bmatrix}
$$

#### Measurement Model:
The GPS sensor provides position measurements, so the measurement model is:
$$
\mathbf{z}_{k} = h(\mathbf{x}_{k}) = \begin{bmatrix}
p_x \\
p_y
\end{bmatrix}
$$


This is the result of applying extended kalman filter on the GPS sensor.

![EKF](https://github.com/user-attachments/assets/2ca3e978-2fe9-4205-8fae-34d9436ff018)


### Conclusion
The **Extended Kalman Filter** extends the power of the standard Kalman Filter to nonlinear systems. It provides a method to estimate the state of a system, even in cases where the motion and sensor models are nonlinear. The recursive nature of EKF makes it ideal for real-time applications in robotics, navigation, and sensor fusion, where continuous updates to state estimates are crucial.



