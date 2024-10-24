#!/usr/bin/python3

# Author @AtsushiSakai 
# These is the reimplementation of the concepts from the original repo @python robotics
# Matplotlib based plotter

import sys
import pathlib

# Append the parent directory to the sys.path
sys.path.append(str(pathlib.Path(__file__).parent.parent))
print(sys.path.append(str(pathlib.Path(__file__).parent.parent)))



import math 
import matplotlib.pyplot as plt 
import numpy as np
# Mpl_toolkits is a library for advanced plotting
from mpl_toolkits.mplot3d import art3d 
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D

from utility.angle import rot_mat_2d


def covariance_ellipse(x, y, cov, chi2=3.0, color="-r", ax=None):
    """
    Plots the covariance matrix in ellipse form.

    Parameters:
    - x (float): X coordinate of the center of the ellipse.
    - y (float): Y coordinate of the center of the ellipse.
    - cov (2x2 array-like): The covariance matrix.
    - chi2 (float, optional): A scalar value that scales the ellipse size.
    - color (str, optional): Color specification for the ellipse.
    - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, the current axes are used.
    """
    covar=cov[0:2,0:2]
    eig_val, eig_vec = np.linalg.eig(covar)
    
    # Ensure eigenvalues are sorted correctly
    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    # Calculate the axes lengths
    a = math.sqrt(chi2 * max(eig_val[big_ind], 0))  # Prevents negative sqrt
    b = math.sqrt(chi2 * max(eig_val[small_ind], 0))  # Prevents negative sqrt

    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    plot_ellipse(x, y, a, b, angle, color=color, ax=ax)

def plot_ellipse(x, y, a, b, angle, color="-r", ax=None, **kwargs):
    """
    Plots an ellipse based on the given parameters.

    Parameters:
    - x (float): X coordinate of the center of the ellipse.
    - y (float): Y coordinate of the center of the ellipse.
    - a (float): Length of the semi-major axis.
    - b (float): Length of the semi-minor axis.
    - angle (float): Orientation of the ellipse in radians.
    - color (str, optional): Color specification for the ellipse.
    - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, the current axes are used.
    """
    
    t = np.linspace(0, 2 * math.pi, 100)  # More points for a smoother ellipse
    px = a * np.cos(t)
    py = b * np.sin(t)
    
    # Rotate the ellipse
    R = rot_mat_2d(angle)
    fx = R @ np.array([px, py])
    px = fx[0, :] + x
    py = fx[1, :] + y
    
    if ax is None:
        plt.plot(px, py, color, **kwargs)
    else:
        ax.plot(px, py, color, **kwargs)
        
        
def arrow(x,y,yaw,arrow_length=1.0,origin_point_plot="xyr",
          head_width=0.1,fc="r",ec="k",**kwargs):
    #Plot an arrow or arrows based on 2D state (x,y,yaw)
    
    if not isinstance(x,float):
        for (i_x,i_y,i_yaw) in zip(x,y,yaw):
            arrow(i_x,i_y,i_yaw,head_width,fc,ec,**kwargs)
            
            
    else: 
        plt.arrow(x,y,
                  arrow_length*math.cos(yaw),
            arrow_length*math.sin(yaw),
            head_width=head_width,
            fc=fc,ec=ec,**kwargs)
        if origin_point_plot is not None:
            plt.plot(x,y,origin_point_plot)
            



class Arrow3D(FancyArrowPatch):
    
    def __init__(self,x,y,z,dx,dy,dz,*args,**kwargs):
        super().__init__((0,0),(0,0),*args,**kwargs)
        self._xyz=(x,y,z)
        self._dxdydz=(dx,dy,dz)
        
        
    def draw(self,render):
        x1,y1,z1=self._xyz
        dx,dy,dz=self._dxdydz
        endx,endy,endz=x1+dx,y1+dy,z1+dz
        xs,ys,zs=proj_transform((x1,endx),(y1,endy),(z1,endz),self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(render)
        
    def do_3d_projectcion(self,render=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def _arrow3D(ax,x,y,z,dx,dy,dz,*args,**kwargs):
        
        arrow=Arrow3D(x,y,z,dx,dy,dz,*args,**kwargs)
        ax.add_artist(arrow)
        
def plot_3d_vector_arrow(ax,p1,p2):
        setattr(Axes3D,'arrow3D',_arrow3D)
        ax.arrow3D(p1[0],p1[1],p1[2],
                   p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2],
                   mutation_scale=20,
                   arrowstyle="-|>", color="r")
        
def plot_triangle(p1,p2,p3,ax):
    ax.add_collection3d(art3d.Poly3DCollection([[p1,p2,p3]],color="b"))



if __name__ == '__main__':
    plot_ellipse(0, 0, 1, 2, np.deg2rad(15))
    plt.axis('equal')
    plt.show()