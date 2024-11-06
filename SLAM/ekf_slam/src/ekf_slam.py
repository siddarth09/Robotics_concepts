#!/usr/bin/python3 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped, Pose,Quaternion
import tf2_ros
import numpy as np
import math

Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2  # Process noise covariance
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2  # Sensor noise
DT = 0.1
STATE_SIZE = 3  # Robot pose [x, y, yaw]
LM_SIZE = 2  # Landmark size [x, y]

class EKFGridBasedSLAM(Node):
    def __init__(self):
        super().__init__('ekf_grid_based_slam')
        
        # Parameters for occupancy grid mapping
        self.declare_parameter('grid_resolution', 0.05)
        self.declare_parameter('grid_size', 100)
        self.declare_parameter('hit_log_odds', 2.0)
        self.declare_parameter('miss_log_odds', -0.5)
        
        # Load parameters
        self.resolution = self.get_parameter('grid_resolution').value
        self.grid_size = self.get_parameter('grid_size').value
        self.hit_log_odds = self.get_parameter('hit_log_odds').value
        self.miss_log_odds = self.get_parameter('miss_log_odds').value

        # Initialize EKF state and occupancy grid
        self.x_est = np.zeros((STATE_SIZE, 1))
        self.P_est = np.eye(STATE_SIZE)
        self.log_odds_grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.origin = [-self.grid_size * self.resolution / 2, -self.grid_size * self.resolution / 2]
        
        # ROS Publishers, Subscribers, and TF broadcaster
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Timer to publish grid and TF at intervals
        self.create_timer(1.0, self.publish_occupancy_grid)
        self.create_timer(0.1, self.publish_map_to_odom_tf)

    def odom_callback(self, msg):
        # Update robot pose based on odometry
        self.x_est[0, 0] = msg.pose.pose.position.x
        self.x_est[1, 0] = msg.pose.pose.position.y
        self.x_est[2, 0] = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        
    def get_yaw_from_quaternion(self,o):
        qx=o.x
        qy=o.y
        qz=o.z
        qw=o.w 
        
        siny_cosp= 2*(qw*qz+qx*qy)
        cosy_cosp= 1- 2* (qy**2+qz**2)
        
        return np.arctan2(siny_cosp,cosy_cosp)
        
    def motion_jacobian(self, u):
        """
        Computes the Jacobian of the motion model with respect to the robot's pose.

        Parameters:
        - u: Control input vector (linear velocity, angular velocity).

        Returns:
        - G: Jacobian matrix for the motion model.
        """
        theta = self.x_est[2, 0]
        v, omega = u[0], u[1]

        if omega == 0:  # Prevent division by zero
            omega = 1e-5

        # State transition matrix G (robot pose part)
        G = np.eye(len(self.x_est))
        G[0, 2] = -v / omega * (np.cos(theta) - np.cos(theta + omega * DT))
        G[1, 2] = -v / omega * (np.sin(theta) - np.sin(theta + omega * DT))

        return G


    def scan_callback(self, msg):
        for i, r in enumerate(msg.ranges):
            if np.isinf(r) or r <= 0:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            x_laser = r * np.cos(angle + self.x_est[2, 0]) + self.x_est[0, 0]
            y_laser = r * np.sin(angle + self.x_est[2, 0]) + self.x_est[1, 0]
            
            # Update EKF with detected landmark
            self.ekf_slam_update(np.array([[x_laser], [y_laser]]))
            
            # Update occupancy grid
            self.update_grid_with_scan(x_laser, y_laser, r)

    def ekf_slam_update(self, z):
        landmark_id = self.data_association(z)

        if landmark_id is None:
            self.add_new_landmark(z)
        else:
            self.update_existing_landmark(z, landmark_id)
            
    def num_landmarks(self):
        """
        Returns the number of landmarks currently in the state vector.

        Returns:
        - int: Number of landmarks.
        """
        # Subtract the robot's pose size and divide by landmark size to get count
        return (len(self.x_est) - STATE_SIZE) // LM_SIZE


    def data_association(self, z):
        # Check distance between the observed landmark and existing landmarks
        min_dist = float('inf')
        min_id = None
        for i in range(self.num_landmarks()):
            lm = self.get_landmark_position_from_state(i)
            delta = lm - self.x_est[0:2, 0]
            q = delta.T @ delta
            if q < min_dist:
                min_dist = q
                min_id = i

        # Return None if no association found (using a threshold)
        threshold = 2.0  # Threshold for Mahalanobis distance
        if min_dist > threshold:
            return None
        return min_id
    
    
    def get_landmark_position_from_state(self, landmark_id):
        """
        Retrieves the position of a given landmark from the state vector.
        
        Parameters:
        - landmark_id: The index of the landmark in the state vector.

        Returns:
        - A numpy array with the x, y position of the landmark.
        """
        # Calculate the start index of the landmark in the state vector
        start_idx = STATE_SIZE + landmark_id * LM_SIZE  # STATE_SIZE is the robot's pose size

        # Retrieve and return the x, y position of the landmark
        lm_x = self.x_est[start_idx, 0]
        lm_y = self.x_est[start_idx + 1, 0]
        return np.array([lm_x, lm_y])

    def add_new_landmark(self, z):
        # Add new landmark to the state vector and covariance matrix
        lm_x, lm_y = z[0, 0], z[1, 0]
        new_lm_state = np.array([[lm_x], [lm_y]])
        self.x_est = np.vstack((self.x_est, new_lm_state))
        self.P_est = np.block([
            [self.P_est, np.zeros((self.P_est.shape[0], LM_SIZE))],
            [np.zeros((LM_SIZE, self.P_est.shape[1])), np.eye(LM_SIZE) * 1.0]
        ])

    
    def compute_jacobian(self, delta, q, landmark_id):
        """
        Computes the Jacobian matrix H of the measurement model.
        
        Parameters:
        - delta: Difference between the landmark and robot positions.
        - q: Squared distance between the landmark and robot.
        - landmark_id: ID of the landmark.

        Returns:
        - H: Jacobian matrix of the measurement model.
        """
        sqrt_q = np.sqrt(q)
        dx, dy = delta[0], delta[1]
        
        # Create Fx_k, the state transition matrix for the landmark
        Fx_k = np.zeros((5, len(self.x_est)))
        Fx_k[0:3, 0:3] = np.eye(3)
        Fx_k[3:5, STATE_SIZE + landmark_id * LM_SIZE:STATE_SIZE + (landmark_id + 1) * LM_SIZE] = np.eye(2)

        # Measurement model Jacobian (H)
        H = np.array([
            [-sqrt_q * dx, -sqrt_q * dy, 0, sqrt_q * dx, sqrt_q * dy],
            [dy, -dx, -q, -dy, dx]
        ]) / q

        return H @ Fx_k
    
    
    def create_quaternion_from_yaw(self, yaw):
        """
        Converts a yaw angle into a quaternion.

        Parameters:
        - yaw: Yaw angle in radians.

        Returns:
        - Quaternion object representing the yaw.
        """
        
        q = Quaternion()
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q


    def update_existing_landmark(self, z, landmark_id):
        # Calculate the innovation and Kalman gain
        lm_pos = self.get_landmark_position_from_state(landmark_id)
        delta = lm_pos - self.x_est[:2, 0]
        q = delta.T @ delta
        H = self.compute_jacobian(delta, q, landmark_id)
        
        # Calculate the Kalman gain
        K = self.P_est @ H.T @ np.linalg.inv(H @ self.P_est @ H.T + Q_sim)
        
        # Calculate the innovation y for the measurement z as a column vector
        innovation = (z - self.x_est[:2, 0].reshape(2, 1))  # Shape (2, 1)

        # Update state and covariance
        self.x_est += K @ innovation  # This directly updates only the relevant parts
        self.P_est = (np.eye(len(self.x_est)) - K @ H) @ self.P_est


        
    def world_to_grid(self, x, y):
        """
        Converts world coordinates to grid indices.

        Parameters:
        - x: X-coordinate in the world frame.
        - y: Y-coordinate in the world frame.

        Returns:
        - (grid_x, grid_y): Coordinates in the occupancy grid.
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y


    def publish_map_to_odom_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = self.x_est[0, 0]
        t.transform.translation.y = self.x_est[1, 0]
        t.transform.rotation = self.create_quaternion_from_yaw(self.x_est[2, 0])
        self.tf_broadcaster.sendTransform(t)

    def publish_occupancy_grid(self):
        np.clip(self.log_odds_grid, -10, 10, out=self.log_odds_grid)
        occupancy_data = (1 - 1 / (1 + np.exp(self.log_odds_grid))) * 100
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = 'map'
        occupancy_grid_msg.info.resolution = self.resolution
        occupancy_grid_msg.info.width = self.grid_size
        occupancy_grid_msg.info.height = self.grid_size
        occupancy_grid_msg.info.origin.position.x = self.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.origin[1]
        occupancy_grid_msg.data = occupancy_data.flatten().astype(int).tolist()
        self.occupancy_grid_pub.publish(occupancy_grid_msg)
        
    def bresenham(self, x0, y0, x1, y1):
        """
        Uses Bresenham's line algorithm to find all cells along a line from (x0, y0) to (x1, y1).

        Parameters:
        - x0, y0: Start coordinates.
        - x1, y1: End coordinates.

        Returns:
        - List of (cell_x, cell_y) along the ray.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells


    def update_grid_with_scan(self, x_laser, y_laser, r):
        grid_x, grid_y = self.world_to_grid(x_laser, y_laser)
        robot_grid_x, robot_grid_y = self.world_to_grid(self.x_est[0, 0], self.x_est[1, 0])
        cells_on_ray = self.bresenham(robot_grid_x, robot_grid_y, grid_x, grid_y)
        for cell_x, cell_y in cells_on_ray:
            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                self.log_odds_grid[cell_x, cell_y] += self.miss_log_odds
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.log_odds_grid[grid_x, grid_y] += self.hit_log_odds

  

def main(args=None):
    rclpy.init(args=args)
    node = EKFGridBasedSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
