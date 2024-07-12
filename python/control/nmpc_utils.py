import numpy as np
from scipy.spatial.transform import Rotation as R

def set_mpc_target_pos(NmpcNode, position_pts): # Mock trajectory for testing
    if position_pts is None:
        position_pts = np.array([[0.0, 0.0, - 2.0]] * (NmpcNode.N + 1))  # Example position setpoint
    else:
        position_pts = position_pts

    thrust_constant = 8.54858e-06
    max_rotor_speed = 1000
    max_thrust = thrust_constant * max_rotor_speed**2
    hover_thrust = 0.8 * max_thrust * 4

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
    if NmpcNode.attitude is None:
        ValueError("Attitude is not set")

    current_R = R.from_quat(NmpcNode.attitude, scalar_first=True)

    # acceleration_sp_NED[0] = np.clip(acceleration_sp_NED[0], -1.0, 1.0)
    # acceleration_sp_NED[1] = np.clip(acceleration_sp_NED[1], -1.0, 1.0)

    if acceleration_sp_NED.shape == (3, 1):
        acceleration_sp_NED = acceleration_sp_NED.reshape(3)
    elif acceleration_sp_NED.shape != (3,):
        raise ValueError("acceleration_sp_NED must be of shape (3,) or (3, 1)")

    acceleration_sp_body = current_R.apply(acceleration_sp_NED)
    print(acceleration_sp_body)

    thrust = acceleration_sp_body[2] * 2.0
    # if np.abs(thrust) < 0.1:
    #    raise ValueError("Thrust is too low")
    thrust = normalize_thrust(thrust)
    thrust = np.clip(thrust, -1.0, 0.0)
    yaw_sp = yaw_sp

    # Mellinger 2011
    t = np.array([acceleration_sp_NED[0], acceleration_sp_NED[1], acceleration_sp_NED[2]])

    z_B = t / np.linalg.norm(t)
    y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
    x_B = np.cross(y_C, z_B) / np.linalg.norm(np.cross(y_C, z_B))
    if (z_B[2] < 0.0):
        x_B = -x_B

    if (np.abs(z_B[2]) < 1e-6):
        x_B = np.empty(3)
        x_B[2] = 1.0

    x_B = x_B / np.linalg.norm(x_B) # normalization
    y_B = np.cross(z_B, x_B)

    R_d = np.column_stack([x_B, y_B, z_B])
    q_d = R.from_matrix(R_d).as_quat(scalar_first=True)
    q_d = [1.0, 0.0, 0.0, 0.0]
    return thrust, q_d
