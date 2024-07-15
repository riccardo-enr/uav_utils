import numpy as np
from math import sqrt, atan2, asin, pi
import scipy
from scipy.spatial.transform import Rotation as R
# import quaternion as quat

def set_mpc_target_pos(NmpcNode, position_pts): # Mock trajectory for testing
    if position_pts is None:
        position_pts = np.array([[0.0, 0.0, - 2.0]] * (NmpcNode.N + 1))  # Example position setpoint
    else:
        position_pts = position_pts

    thrust_constant = 8.54858e-06
    max_rotor_speed = 1000
    max_thrust = thrust_constant * max_rotor_speed**2
    hover_thrust = 0.5 * max_thrust * 4

    acc_setpoint = np.array([0.0, 0.0, hover_thrust / 2.0])

    for j in range(NmpcNode.solver.N):
        yref = np.concatenate([position_pts[j], np.zeros(3), acc_setpoint], axis=0)
        NmpcNode.p[-9:] = yref
        NmpcNode.solver.set(j, "p", NmpcNode.p)

def set_current_state(NmpcNode):
    """aggregates individual states to combined state of system
    """
    # , NmpcNode.state_timestamp
    NmpcNode.current_state = np.concatenate((NmpcNode.position, NmpcNode.velocity, NmpcNode.acceleration), axis=None)

    #self.imu_data = np.concatenate((self.linear_accel_real, self.angular_accel_real, self.imu_timestamp), axis=None)

    if NmpcNode.state_history is None:
        NmpcNode.state_history = []

    NmpcNode.state_history.append(NmpcNode.current_state)
    #self.imu_history.append(self.imu_data)

def thrust_to_motor_values(thrust):
    """Converts thrust to motor values
    """
    thrust_constant = 8.54858e-06
    max_rotor_speed = 1000

    # max_thrust
    max_thrust = thrust_constant * max_rotor_speed**2
    u_motor_values = thrust / max_thrust
    # u_motor_values = - u_motor_values

    max_input = 1.0
    min_input = 0.1

    for i in range(len(u_motor_values)):
        if u_motor_values[i] > max_input:
            u_motor_values[i] = max_input
        elif u_motor_values[i] < min_input:
            u_motor_values[i] = min_input

    return u_motor_values

def motor_values_to_thrust(motor_values):
    """Converts motor values to thrust
    """
    thrust_constant = 8.54858e-06
    max_rotor_speed = 1000

    motor_values /= max_rotor_speed

    # max_thrust
    max_thrust = thrust_constant * max_rotor_speed**2
    thrust = motor_values * max_thrust

    return thrust

def normalize_thrust(thrust):
    """Normalizes thrust to be between 0 and 1
    """
    thrust_constant = 8.54858e-06
    max_rotor_speed = 1000

    # max_thrust
    max_thrust = thrust_constant * max_rotor_speed**2
    max_thrust *= 4
    thrust = thrust / max_thrust

    return thrust

def acceleration_sp_to_thrust_q(NmpcNode, acceleration_sp_NED, yaw_sp):
    """
    Converts the desired acceleration in the NED frame to thrust and attitude quaternion.

    Args:
        NmpcNode: The NmpcNode object.
        acceleration_sp_NED: The desired acceleration in the NED frame, from NMPC.
        yaw_sp: The desired yaw angle in the NED frame.

    Returns:
        thrust: The thrust value in the body frame (normalized) [0,-1].
        q_d: The desired attitude quaternion of the UAV body frame in the NED.
    """
    if NmpcNode is None:
        current_attitude = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        current_attitude = NmpcNode.attitude

    current_R = quat2RotMatrix(current_attitude)

    normal_acc = False

    max_acc_xy = max(np.abs(acceleration_sp_NED[0]), np.abs(acceleration_sp_NED[1]))
    if normal_acc:
        for i in range(len(acceleration_sp_NED) - 1):
            if np.abs(acceleration_sp_NED[i]) > 1.0:
                acceleration_sp_NED[i] = acceleration_sp_NED[i] / max_acc_xy

    if acceleration_sp_NED.shape == (3, 1):
        acceleration_sp_NED = acceleration_sp_NED.reshape(3)
    elif acceleration_sp_NED.shape != (3,):
        raise ValueError("acceleration_sp_NED must be of shape (3,) or (3, 1), instead the shape is " + str(acceleration_sp_NED.shape) + ".")

    acceleration_sp_body = current_R @ acceleration_sp_NED

    thrust = acceleration_sp_body[2] * NmpcNode.mass

    q_d, eul_d = acc2quaternion(acceleration_sp_body, yaw_sp, thrust, euler=True)
    thrust = normalize_thrust(thrust)
    thrust = np.clip(thrust, -1.0, 0.0)

    return thrust, q_d, eul_d

def acc2quaternion(acc_sp, yaw, thrust, euler=False):
    """
    Converts the desired acceleration in the NED frame to the desired attitude quaternion of the UAV body frame in the NED.

    Args:
        acc_sp: The desired acceleration in the NED frame.
        yaw: The desired yaw angle in the NED frame.

    Output:
        q_d: The desired attitude quaternion of the UAV body frame in the NED.
    """
    g_ = 9.81
    acc_sp = np.array([acc_sp[0], acc_sp[1], acc_sp[2]])

    if scipy.linalg.norm(acc_sp) < 1e-6 or acc_sp[2] + g_ > 0.0:
        z_B = np.array([0.0, 0.0, 1.0])
    else:
        z_B = - acc_sp / np.linalg.norm(acc_sp)

    x_C = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_C = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    x_B = np.cross(y_C, z_B)
    x_B /= np.linalg.norm(x_B)

    if z_B[2] < 0.0:
        x_B = -x_B

    if np.abs(z_B[2]) < 1e-6:
        x_B = np.array([0.0, 0.0, 1.0])

    y_B = np.cross(z_B, x_B)
    y_B /= np.linalg.norm(y_B)

    R_d = np.column_stack([x_B, y_B, z_B])
    # print(R_d)
    q_d = rot2Quaternion(R_d)

    acc_sp_check = R_d @ np.array([0.0, 0.0, thrust])

    if not np.allclose(acc_sp, acc_sp_check):
        print("Error in acc2quaternion: acc_sp = ", acc_sp, ", acc_sp_check = ", acc_sp_check)

    if euler:
        eul_d = R.from_quat(q_d).as_euler('zyx', degrees=False)
        return q_d, eul_d
    else:
        return q_d

def quat2RotMatrix(q):
    rotmat = np.zeros((3, 3))
    rotmat[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    rotmat[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    rotmat[0, 2] = 2 * q[0] * q[2] + 2 * q[1] * q[3]

    rotmat[1, 0] = 2 * q[0] * q[3] + 2 * q[1] * q[2]
    rotmat[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    rotmat[1, 2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]

    rotmat[2, 0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    rotmat[2, 1] = 2 * q[0] * q[1] + 2 * q[2] * q[3]
    rotmat[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

    return rotmat

def rot2Quaternion(R):
    quat = np.zeros(4)
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0  # S=4*qw
        quat[0] = 0.25 * S
        quat[1] = (R[2, 1] - R[1, 2]) / S
        quat[2] = (R[0, 2] - R[2, 0]) / S
        quat[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S=4*qx
        quat[0] = (R[2, 1] - R[1, 2]) / S
        quat[1] = 0.25 * S
        quat[2] = (R[0, 1] + R[1, 0]) / S
        quat[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S=4*qy
        quat[0] = (R[0, 2] - R[2, 0]) / S
        quat[1] = (R[0, 1] + R[1, 0]) / S
        quat[2] = 0.25 * S
        quat[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S=4*qz
        quat[0] = (R[1, 0] - R[0, 1]) / S
        quat[1] = (R[0, 2] + R[2, 0]) / S
        quat[2] = (R[1, 2] + R[2, 1]) / S
        quat[3] = 0.25 * S
    return quat

def quaternion_multiply(q1, q0):
    # TODO: Spostare in quaternion.py
    Q1 = np.array([
        [q1[0], -q1[1], -q1[2], -q1[3]],
        [q1[1], q1[0], -q1[3], q1[2]],
        [q1[2], q1[0], q1[0], -q1[1]],
        [q1[3], -q1[2], q1[1], q1[0]]
    ])

    return Q1 @ q0

def skew_simmetric(z):
    return np.array([
        [0, -z[2], z[1]],
        [z[2], 0, -z[0]],
        [-z[1], z[0], 0]
    ])

def s_function(a, b):
    a = np.array(a)
    b = np.array(b)

    a = np.reshape(a, (3, 1))
    b = np.reshape(b, (3, 1))

    out = np.sqrt(1 - (np.dot(np.transpose(a), b))**2)
    return out
