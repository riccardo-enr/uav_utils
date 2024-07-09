import numpy as np
from math import sqrt
from scipy.spatial.transform import Rotation as R
from quaternion import quaternion_multiply, euler_to_quaternion_numpy

def NED_to_ENU(input_array):
    """
    Convert NED frame convention from drone into ENU frame convention used in MPC

    Parameters:
    input_array (np.ndarray): The input to be rotated. Should be a NumPy array of shape (3,) for a 3D vector or (4,) for a quaternion.

    Returns:
    np.ndarray: The rotated 3D vector or quaternion.
    """
    
    if input_array.shape == (3,):
        # Handle as a 3D vector
        # A 180-degree rotation around the x-axis flips the signs of the y and z components
        rot_x = R.from_euler('x', 180, degrees=True)
        rotated_array = rot_x.apply(input_array)
        
        #rotated_array = rot_z.apply(rotated_array)
    elif input_array.shape == (4,):
        
        # Handle as a quaternion
        # A 180-degree rotation around the x-axis flips the signs of the y and z components
        
        input_array = input_array/np.linalg.norm(input_array)
        rotated_array = np.zeros(4)
        
        rotated_array[0] = input_array[0]
        rotated_array[1] = input_array[1]
        rotated_array[2] = -input_array[2]
        rotated_array[3] = -input_array[3]        
        
    else:
        raise ValueError("Input array must be either a 3D vector or a quaternion (shape (3,) or (4,)).")
    
    return rotated_array

def quat_NED2ENU(input_quat):
    """
    Convert a quaternion from NED (North-East-Down) to ENU (East-North-Up) frame.

    This function converts a quaternion representing a rotation in the NED coordinate frame to
    the equivalent quaternion in the ENU coordinate frame.

    Parameters:
    - input_quat: A numpy array representing the input quaternion in NED frame [w, x, y, z]

    Returns:
    - output_quat: A numpy array representing the converted quaternion in ENU frame [w, x, y, z]
    """
    ned2enu_quaternion = euler_to_quaternion_numpy([np.pi, 0, np.pi/2])
    
    output_quat = quaternion_multiply(ned2enu_quaternion, input_quat)
    
    return output_quat

def quat_ENU2NED(input_quat):
    """
    Convert a quaternion from ENU (East-North-Up) to NED (North-East-Down) frame.

    This function converts a quaternion representing a rotation in the ENU coordinate frame to
    the equivalent quaternion in the NED coordinate frame.

    Parameters:
    - input_quat: A numpy array representing the input quaternion in ENU frame [w, x, y, z]

    Returns:
    - output_quat: A numpy array representing the converted quaternion in NED frame [w, x, y, z]
    """
    enu2ned_quaternion = euler_to_quaternion_numpy([-np.pi, 0, -np.pi/2])
    
    output_quat = quaternion_multiply(enu2ned_quaternion, input_quat)
    
    return output_quat