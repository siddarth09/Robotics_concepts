# EKF SLAM (Extended Kalman Filter SLAM)

## Overview
EKF SLAM is a technique used to estimate both the robot's trajectory and the map of its environment in simultaneous localization and mapping (SLAM) problems. The core idea behind EKF SLAM is to use an Extended Kalman Filter (EKF) to fuse sensor measurements (like odometry and laser scans) with the robot's state estimate and the map of the environment.

The key idea is that the robot not only estimates its position but also builds a map by identifying landmarks. As the robot moves, it updates its knowledge of both its own state (position, orientation) and the locations of the landmarks.

## EKF SLAM Steps

1. **Prediction Step** (Motion Model):
    - The robot's state is predicted based on its motion model, using control inputs (like velocity and steering).
    - The state prediction is given by:

    \[
    \hat{x}_k^- = f(\hat{x}_{k-1}, u_k) = 
    \begin{bmatrix}
        x_{k-1} + \Delta t \cdot v_k \cdot \cos(\theta_{k-1}) \\
        y_{k-1} + \Delta t \cdot v_k \cdot \sin(\theta_{k-1}) \\
        \theta_{k-1} + \Delta t \cdot \omega_k
    \end{bmatrix}
    \]

    where:
    - \(x, y, \theta\) are the robot's position (x, y) and orientation (\(\theta\)),
    - \(v_k\) is the robot's velocity,
    - \(\omega_k\) is the angular velocity,
    - \(\Delta t\) is the time step.

    - The covariance prediction is:

    \[
    P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q_k
    \]

    where:
    - \(P_k^-\) is the predicted covariance matrix,
    - \(F_{k-1}\) is the Jacobian of the motion model with respect to the state,
    - \(Q_k\) is the process noise covariance.

2. **Update Step** (Measurement Model):
    - When a new observation (e.g., a laser scan) is made, the map is updated.
    - The expected measurement based on the predicted state and landmarks is given by:

    \[
    z_k = h(\hat{x}_k^-)
    \]

    where \(h(\hat{x}_k^-)\) is the expected measurement function, which could be derived from a sensor model.
    
    - The innovation or residual is:

    \[
    \tilde{z}_k = z_k - h(\hat{x}_k^-)
    \]
    
    - The Kalman gain is computed to update the state estimate:

    \[
    K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}
    \]

    where:
    - \(H_k\) is the Jacobian of the measurement model with respect to the state,
    - \(R_k\) is the measurement noise covariance.

    - The updated state estimate is:

    \[
    \hat{x}_k = \hat{x}_k^- + K_k \tilde{z}_k
    \]

    - The updated covariance is:

    \[
    P_k = (I - K_k H_k) P_k^-
    \]

## EKF SLAM with Landmarks

The state vector in EKF SLAM includes both the robot's state (position and orientation) and the positions of the landmarks. If the robot has \(N\) landmarks, the state vector \(\hat{x}_k\) is:

\[
\hat{x}_k = \begin{bmatrix} \hat{x}_k^r \\ \hat{x}_k^m \end{bmatrix}
\]

where:
- \(\hat{x}_k^r = [x_k, y_k, \theta_k]^T\) is the robot's state.
- \(\hat{x}_k^m = [x_1, y_1, \dots, x_N, y_N]^T\) are the landmark positions.

In the prediction step, the robot’s state is updated as before, but the landmark positions remain unchanged unless updated by new measurements. The measurement update also involves both the robot's state and the landmarks.

### Measurement Update

When a new measurement for a landmark is available, the update step involves calculating the difference between the predicted and actual landmark positions. If the robot’s position \(\hat{x}_k^r\) and the landmark’s position \(\hat{x}_k^m\) are known, the expected measurement \(z_k\) for a landmark \(m\) at position \([x_m, y_m]\) can be computed as:

\[
z_k = \begin{bmatrix}
    r_k \\
    \phi_k
\end{bmatrix} = \begin{bmatrix}
    \sqrt{(x_m - x_k)^2 + (y_m - y_k)^2} \\
    \text{atan2}(y_m - y_k, x_m - x_k)
\end{bmatrix}
\]

where:
- \(r_k\) is the range to the landmark,
- \(\phi_k\) is the bearing to the landmark.

## Key Assumptions

1. **Non-linearities**: The EKF assumes that the motion model and measurement model are non-linear functions of the state.
2. **Gaussian Noise**: The EKF assumes Gaussian noise in the process and measurement models.
3. **Linearization**: The EKF linearizes the non-linear models using first-order Taylor expansion (Jacobians).

## Conclusion

EKF SLAM is a widely used technique for mobile robots to simultaneously localize themselves and map the environment. By combining motion and sensor data, the EKF provides estimates of both the robot’s state and the map of landmarks. It uses the Kalman filter framework to predict and update the robot’s state in a probabilistic manner.

### WORKING 

![Untitled design](https://github.com/user-attachments/assets/aa367db3-c43e-4224-9dd2-825f136b6858)

