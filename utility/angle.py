# Authour @AtsushiSakai 
# angle helper functions for transformation

import numpy as np
from scipy.spatial.transform import Rotation as Rot 

def rot_mat_2d(angle):
    """
    Create 2d Rotation matrix from angle 

    Args:
        angle : 
        
    returns: 
        A 2d rotation matrix
        
    """
    return Rot.from_euler ('z',angle).as_matrix()[0:2,0:2]


def angle_mod(x,zero_2_2pi=False, degree=False):
    """Angle modulo operation
    
    Default angle modulo range is [-pi,pi)
    
    Args:
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    returns:
    ret : float or ndarray
        an angle or an array of modulated angle.
    """
    if isinstance(x,float):
        is_float=True
    else: 
        is_float=False
        

    x=np.asarray(x).flatten()
    
    if degree:
        x=np.deg2rad(x)
        
    if zero_2_2pi:
        mod_angle= x%(2*np.pi)
        
    else:
        # [-pi to pi)
        mod_angle= (x+np.pi)%(2*np.pi)-np.pi 
        
        
    if degree:
        mod_angle=np.rad2deg(mod_angle)
        
    if is_float:
        return mod_angle.item()
    
    else:
        
        return mod_angle
    