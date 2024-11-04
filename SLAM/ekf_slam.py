import sys 
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np 
import math 
import  matplotlib.pyplot  as plt
from utility.angle import angle_mod


#EKF State covariance 

Cx= np.diag([0.5,0.5,np.deg2rad(30.0)])


#Simulation params

Q_sim=np.diag([0.2,np.deg2rad(1.0)])**2
R_sim=np.diag([1.0,np.deg2rad(10.0)])**2 

DT=0.1
SIM_TIME=50.0
MAX_RANGE=20.0
MAHALANOBIS_DIST=2.0 #THRESHOLD FOR MAHALONOBIS DISTANCE
STATE_SIZE=3 #STATE SPACE (X,Y,YAW)
CONTROL_SIZE=2 #CONTROL STATE (X,Y)



def calculate_input():
    v=1.0 #m/s
    yaw_rate=0.1 #[rad/s]
    u=np.array([[v,yaw_rate]]).T
    return u 


def motion_model(x,u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    
    return x



def observation(xTrue,xd,u,landmarks):
    x_pred = motion_model(xTrue,u)
    
    z= np.zeros ((0,3))
    
    for i in range(len(landmarks[:,0])):
        dx=landmarks[i,0]-x_pred[0,0]
        dy=landmarks[i,1]-x_pred[1,0]
        d = math.hypot(dx,dy)
        angle=angle_mod(math.atan2(dy,dx)-x_pred[2,0])
        if d<=MAX_RANGE:
            noise=d+np.random.randn()*Q_sim[0,0]**0.5 
            angle_with_noise=angle+np.random.randn()*Q_sim[1,1]**0.5
            measurement_i=np.array([noise,angle_with_noise,i])
            z=np.vstack((z,measurement_i))
            
    new_control=np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    
    xd=motion_model(xd,new_control)
    
    
    return x_pred,z,xd,new_control


def calculate_n_lm(x):
    n=int((len(x)-STATE_SIZE)/CONTROL_SIZE)
    return n


def G(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros((STATE_SIZE, CONTROL_SIZE * calculate_n_lm(x)))))
    
    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)
    
    g = np.eye(len(x)) + Fx.T @ jF @ Fx
    
    return g, Fx

def calc_landmark_position(x,z):
    
    measurement=np.zeros((2,1))
    measurement[0,0]=x[0,0]+z[0]*np.cos(x[2,0]+z[1])
    measurement[1,0]=x[1,0]+z[0]*np.sin(x[2,0]+z[1])
    
    return measurement

def get_landmark_pose_from_state(x,ind):
    lm=x[STATE_SIZE+CONTROL_SIZE*ind:STATE_SIZE+CONTROL_SIZE*(ind+1),:]
    return lm

def search_correspond_landmark_id(xAug,Paug,zi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calculate_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_pose_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, Paug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y)

    min_dist.append(MAHALANOBIS_DIST)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id

def calc_innovation(lm,x_est,p_est,z,Lmid):
    delta=lm-x_est[0:2]
    q=(delta.T @ delta)[0,0]
    z_angle=np.arctan2(delta[1,0],delta[0,0]-x_est[2,0])
    zp=np.array([[math.sqrt(q),angle_mod(z_angle)]])
    y=(z-zp).T
    h=H(q,delta,x_est,Lmid+1)
    S=h@p_est@h.T+Cx[0:2,0:2]
    
    return y,S,h

def H(q,delta,x,i):
    sq=math.sqrt(q)
    g=np.array([[-sq * delta[0, 0], 
                 - sq * delta[1, 0],
                 0, 
                 sq * delta[0, 0], 
                 sq * delta[1, 0]],
                
                [delta[1, 0], - delta[0, 0],
                 - q, - delta[1, 0], delta[0, 0]]])
    g=g/q 
    
    nLM=calculate_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))
    
    F=np.vstack((F1,F2))
    
    h=g@F 
    
    return h


def ekf_slam (x_est,p_est,u,z):
    #predict
    g,Fx=G(x_est,u)
    x_est[0:STATE_SIZE]=motion_model(x_est[0:STATE_SIZE],u)
    p_est= g.T @ p_est @ g + Fx.T @ Cx @ Fx 
    initP = np.eye(2)
    
    
    #update 
    
    for iz in range(len(z[:,0])):
        min_id = search_correspond_landmark_id(x_est,p_est,z[iz,0:2])
        
        nLm= calculate_n_lm(x_est)
        if min_id == nLm:
            
            print("New landmark")
            #Extend State and covariance matrix 
            
            xAug = np.vstack((x_est, calc_landmark_position(x_est, z[iz, 0:2])))

            pAug = np.vstack((np.hstack((p_est, np.zeros((len(x_est), CONTROL_SIZE)))),
                              np.hstack((np.zeros((CONTROL_SIZE, len(x_est))), initP))))
            
            x_est=xAug
            p_est=pAug
            
        landmark=get_landmark_pose_from_state(x_est,min_id)
        y,S,H =calc_innovation(landmark,x_est,p_est,z[iz,0:2],min_id)
        
        
        K=(p_est @ H.T) @ np.linalg.inv(S)
        x_est=x_est+(K @ y)
        
        p_est = (np.eye(len(x_est))- (K @ H)) @ p_est
        
    x_est[2]=angle_mod(x_est[2])
    
    return x_est,p_est            
    
def main(show_animation):
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calculate_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calculate_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == "__main__":
    main(show_animation=True)

            
        
        
