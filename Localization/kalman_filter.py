import math 
import numpy as np
import matplotlib as plt


class KalmanFilter():
    def __init__(self,state_space, measurement, dt, control=0):
        self.X = np.zeros((state_space,1)) #State vector (initial) (4x1)
        print(f"state \n {self.X}")
        self.P = np.eye(state_space) #Covariance Matrix
        print(f"Covariance matrix of state \n {self.P}")

        #State transition Matrix 
        self.A = np.eye(state_space)
        for i in range (state_space//2):
            self.A[i,i+state_space//2]=dt
        print(f"Transition state matrix \n {self.A}")

        #Control matrix 
        self.B = np.zeros((state_space,control))
        print(f"Control matrix \n {self.B}")

        #Measurement Matrix (H)
        self.H = np.zeros((measurement,state_space))
        for i in range (measurement):
            self.H[i,i]=1
            
        print(f"Measurement matrix \n {self.H}")

        #Process Noise covar (Q)

        self.Q=np.eye(state_space)*0.1

        self.R=np.eye(measurement)*0.1

        #Identity
        self.I= np.eye(state_space)


    def predict(self,U=np.zeros((1,1))):
            
            # Finding State and covariance (predicted)
            self.X = np.dot(self.A,self.X)
            self.P = np.dot(np.dot(self.A,self.P),np.matrix.transpose(self.A)) + self.Q

            return self.X
        
    def update(self,Z):
            
            #Calculate kalman gain (K_k)
            Y=Z- np.dot (self.H,self.X)
            S=np.dot(np.dot(self.H,self.P),np.matrix.transpose(self.H))+self.R
            K = np.dot(np.dot(self.P,np.matrix.transpose(self.H)),np.linalg.inv(S))

            self.X=self.X+ np.dot(K,Y)
            self.P=np.dot(self.I-np.dot(K,self.H),self.P)

            return self.X


def main():

    dt= 1.0

    kalman=KalmanFilter(4,2,dt,0)

    measurement_m= [
        np.array([[1.0],[2.0]]),
        np.array([[3.0],[4.4]]),
        np.array([[4.8],[3.0]],)

    ]

    for Z in measurement_m:
        kalman.predict()
        X_updated=kalman.update(Z)

        print(f"Updated state estimation\n {X_updated}")


if __name__=="__main__":
    main()