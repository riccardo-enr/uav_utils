import numpy as np
import casadi as cs
from scipy.spatial.transform import Rotation as R
# from casadi import SX, MX, vertcat, Function, sqrt, norm_2, dot, cross, atan2, if_else
import spatial_casadi as sc

def euler_to_quaternion_numpy(rpy):
    """
    Convert Euler angles to quaternion.

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    np.ndarray
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()
    
def quat2rotm(q):
    """
    Convert quaternion to rotation matrix

    Parameters:
    - q (list): Quaternion represented as a list [qw, qx, qy, qz]

    Returns:
    - np.array: Rotation matrix
    """
    pass
    
def quat2rotm(qw: cs.MX.sym, qx: cs.MX.sym, qy: cs.MX.sym, qz: cs.MX.sym) -> cs.MX.sym:
    """
    Convert quaternion to rotation matrix (Casadi version)

    Parameters:
    - qw (cs.MX.sym): Scalar component of the quaternion
    - qx (cs.MX.sym): First vector component of the quaternion
    - qy (cs.MX.sym): Second vector component of the quaternion
    - qz (cs.MX.sym): Third vector component of the quaternion

    Returns:
    - cs.MX.sym: Rotation matrix

    Example usage:
    ```
    qw = cs.MX.sym('qw')
    qx = cs.MX.sym('qx')
    qy = cs.MX.sym('qy')
    qz = cs.MX.sym('qz')
    R = quat2rotm(qw, qx, qy, qz)
    ```
    """
    r11 = 2 * (qw ** 2 + qx ** 2) - 1
    r12 = 2 * (qx * qy - qw * qz)
    r13 = 2 * (qx * qz + qw * qy)
    r21 = 2 * (qx * qy + qw * qz)
    r22 = 2 * (qw ** 2 + qy ** 2) - 1
    r23 = 2 * (qy * qz - qw * qx)
    r31 = 2 * (qx * qz - qw * qy)
    r32 = 2 * (qy * qz + qw * qx)
    r33 = 2 * (qw ** 2 + qz ** 2) - 1

    R = cs.MX.sym("R", 3, 3)
    R[0, 0] = r11
    R[0, 1] = r12
    R[0, 2] = r13
    R[1, 0] = r21
    R[1, 1] = r22
    R[1, 2] = r23
    R[2, 0] = r31
    R[2, 1] = r32
    R[2, 2] = r33
    return R

def quaternion_product_numpy(q1, q2):
    """
    Compute the Hamilton product of two quaternions.

    This function calculates the Hamilton product of two quaternions using the formula:
    q = q1 * q2 = (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                   a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                   a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                   a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)

    Parameters:
    - q1: A numpy array representing the components of the first quaternion (a1, b1, c1, d1)
    - q2: A numpy array representing the components of the second quaternion (a2, b2, c2, d2)

    Returns:
    - A numpy array representing the components of the Hamilton product quaternion (a, b, c, d)
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    out_quat = np.array([a, b, c, d])

    return out_quat

def quaternion_product_casadi(q1, q2):
    """
    Compute the Hamilton product of two quaternions using Casadi.

    Parameters:
    - q1: MX, the first quaternion represented as a 4x1 vector [a1, b1, c1, d1]
    - q2: MX, the second quaternion represented as a 4x1 vector [a2, b2, c2, d2]

    Returns:
    - MX: The Hamilton product quaternion as a 4x1 vector
    """
    a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
    a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
    
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return cs.vertcat(a, b, c, d)

def eul2quat(rpy):
    """
    Convert Euler angles to quaternion.

    Parameters:
    rpy : np.ndarray roll, pitch, yaw

    Returns:
    np.ndarray
        Quaternion [w, x, y, z] representing the rotation.
    """
    roll, pitch, yaw = rpy
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to quaternion (scalar-last format)
    q = r.as_quat()
    
def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)
    
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])