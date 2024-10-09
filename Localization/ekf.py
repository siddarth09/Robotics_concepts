import numpy as np
import matplotlib.pyplot as plt

class ExtendedKF():
    def __init__(self, state_dim, measurement_dim, control_dim, dt):
        self.X = np.zeros((state_dim, 1))  # State vector (px, py, vx, vy)
        self.P = np.eye(state_dim)  # Covariance matrix
        self.Q = np.eye(state_dim) * 0.1  # Process noise covariance
        self.R = np.eye(measurement_dim) * 0.5  # Measurement noise covariance
        self.I = np.eye(state_dim)  # Identity matrix
        self.dt = dt  # Time step

    def g(self, X, U):
        # Motion model: state transition
        px, py, vx, vy = X.flatten()
        return np.array([[px + vx * self.dt],
                         [py + vy * self.dt],
                         [vx],
                         [vy]])

    def h(self, X, landmarks):
        # Measurement model: distance to landmarks
        px, py = X[0, 0], X[1, 0]
        distances = [np.sqrt((px - lm[0]) ** 2 + (py - lm[1]) ** 2) for lm in landmarks]
        return np.array(distances).reshape(-1, 1)

    def jacobian_G(self):
        # Jacobian of the state transition function
        return np.array([[1, 0, self.dt, 0],
                         [0, 1, 0, self.dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def jacobian_H(self, X, landmarks):
        # Jacobian of the measurement function
        px, py = X[0, 0], X[1, 0]
        jacobian = []
        
        for lm in landmarks:
            lx, ly = lm
            dist = np.sqrt((px - lx) ** 2 + (py - ly) ** 2)
            jacobian.append([ (px - lx) / dist, (py - ly) / dist, 0, 0 ])
        
        return np.array(jacobian)

    def predict(self, U):
        G = self.jacobian_G()
        self.X = self.g(self.X, U)
        self.P = np.dot(np.dot(G, self.P), np.matrix.transpose(G)) + self.Q
        return self.X

    def update(self, Z, landmarks):
        # Use the predicted state for measurement update
        H = self.jacobian_H(self.X, landmarks)
        y = Z - self.h(self.X, landmarks)
        S = np.dot(np.dot(H, self.P), np.matrix.transpose(H)) + self.R
        K = np.dot(np.dot(self.P,np.matrix.transpose(H)), np.linalg.inv(S))
        self.X = self.X + np.dot(K, y)
        self.P = np.dot(self.I - np.dot(K, H), self.P)
        return self.X 

def main():
    ekf = ExtendedKF(4, 2, 2, 0.5)  # dt = 0.5

    # Initial state and covariance
    ekf.X = np.array([[0], [0], [0], [0]])  # Initial position at [0, 0]
   

    # Landmarks from the question 
    landmarks = np.array([[5, 5], [-5, 5]])

    # Storage
    true_trajectory = []
    estimated_trajectory = []
    predicted_trajectory = []  # To store predicted states
    
    # Time and control inputs
    time_total = 40
    for t in np.arange(0, time_total + ekf.dt, ekf.dt):
        if t <= 10:
            U = np.array([[1], [0]])  
        elif t <= 20:
            U = np.array([[0], [-1]])  
        elif t <= 30:
            U = np.array([[-1], [0]])  
        else:
            U = np.array([[0], [1]])  

        
        pose = ekf.g(ekf.X, U)
        true_trajectory.append([pose[0, 0], pose[1, 0]])

        # Predict step
        predicted_state = ekf.predict(U)
        predicted_trajectory.append([predicted_state[0, 0], predicted_state[1, 0]])  

        # Simulated measurement (
        Z = ekf.h(pose, landmarks) + np.random.multivariate_normal([0, 0], ekf.R).reshape(2, 1)

        # Update step
        updated_state = ekf.update(Z, landmarks)
        estimated_trajectory.append([updated_state[0, 0], updated_state[1, 0]])

    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)
    predicted_trajectory = np.array(predicted_trajectory)  

    plt.figure()
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', label='True trajectory')
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'r--', label='Estimated trajectory')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'g--', label='Predicted trajectory')  # Plot predicted state
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='green', label='Landmarks')
    plt.title("True vs Estimated vs Predicted Trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
