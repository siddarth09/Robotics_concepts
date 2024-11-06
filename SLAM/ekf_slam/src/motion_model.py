#!/usr/bin/python3

import rclpy 
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np 
import math 


DT=0.1
STATE_SIZE=3 
LM_SIZE=2


class MotionModel(Node):
    def __init__(self):
        super().__init__('motion_model')
        
        #Subscriber 
        self.odom_sub=self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.scan_sub=self.create_subscription(LaserScan,"/scan",self.scan_callback,10)
        
        self.occupancy_pub=self.create_publisher(OccupancyGrid,"/map",10)
        
        #initialize the pose
        self.state_space=np.zeros((STATE_SIZE,1))
        self.state_covar=np.zeros(STATE_SIZE)
        self.control_input=np.zeros((2,1))
        self.x=0.0
        self.y=0.0
        self.yaw=0.0
        
        
        #Variables pertaining to scan data
        
        self.angle_min= 0.0
        self.angle_max= 0.0 
        
        self.angle_increment= 0.0 #Angular distance between measurements
        
        self.time_between_measurement= 0.0 # s
        
        
        self.prev_time=self.get_clock().now()
        
    def calculate_n_lm(self,x):
        n=int((len(x)-STATE_SIZE)/LM_SIZE)
        return n
    
    def odom_callback(self,msg):
        
        vel_x=msg.twist.twist.linear.x
        yaw=msg.twist.twist.angular.z 
        self.x=msg.pose.pose.position.x
        self.y=msg.pose.pose.position.y
        
       
        present_time=self.get_clock().now()
        dt=(present_time-self.prev_time).nanoseconds/1e9
        self.prev_time=present_time
        self.control_input=np.array([[vel_x,yaw]]).T
        #Updating the position 
        
        odom_covariance=np.array(msg.pose.covariance).reshape((6,6))
        odom_pose_covar=odom_covariance[:3,:3]
        
        self.get_logger().info(f"CONTROL INPUT : {self.control_input}")
        self.get_logger().info(f"covariance: \n {odom_pose_covar}")
        
    def update_motion(self,x,p,u,process_noise):
        """
        
        Updating state space and covariance p with motion model 
        """
        
        F=np.eye(STATE_SIZE)
        G,Fx = self.motion_jacobian(x,u)
        B=np.array([[DT*math.cos(x[2,0]),0],
                    [DT*math.sin(x[2,0]),0],
                    [0.0,DT]])
        x=(F @ x) + (B @ u)
        
        self.get_logger().info(f"New state space: \n {x}")
        
        #Updating the covar 
        p= G.T @ p @ G + Fx.T @process_noise @ Fx 
        self.get_logger().info(f"Updated covariance:\n {p}")        
        
        
        return x,p
    
    
    def motion_jacobian(self,x,u):
        """
        Calculate the jacobian matrix 
        """
        
        Fx= np.hstack((np.eye(STATE_SIZE),np.zeros((STATE_SIZE,2*self.calculate_n_lm(x)))))
        
        
        jF= np.array([[0.0,0.0,-DT*u[0,0]*math.sin(x[2,0])],
                      [0.0,0.0,DT*u[0,0]*math.cos(x[2,0])],
                      [0.0,0.0,0.0]],dtype=float)
        
        G=np.eye(len(x)) + Fx.T @ jF @ Fx 
        
        return G,Fx  
    
    def scan_callback(self,msg):
        """
        Callback for scan data 
        """
        self.angle_max=msg.angle_max 
        self.angle_min=msg.angle_min
        
        
        
        
    
def main():
    rclpy.init()
    ekf_slam_node = MotionModel()
    rclpy.spin(ekf_slam_node)
    ekf_slam_node.destroy_node()
    rclpy.shutdown()

   
if __name__=="__main__":
    main()
   
   
        
