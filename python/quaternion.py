import numpy as np
import casadi as cs

def quat2euler(q):
    """
    Convert quaternion to euler angles

    Parameters:
    - q (list): Quaternion represented as a list [qw, qx, qy, qz]

    Returns:
    - list: Euler angles represented as a list [roll, pitch, yaw]
    """
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    
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

def quaternion_product(a1, b1, c1, d1, a2, b2, c2, d2):
    """
    Compute the Hamilton product of two quaternions using Casadi.

    Parameters:
    - a1, b1, c1, d1: components of the first quaternion
    - a2, b2, c2, d2: components of the second quaternion

    Returns:
    - components of the Hamilton product quaternion
    """
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return a, b, c, d

def quaternion_product(q1, q2):
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