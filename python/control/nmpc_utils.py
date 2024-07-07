import numpy as np

def set_mpc_target_pos(NmpcNode, position_pts : None, attitude_pts : None):
    # Mock trajectory for testing
    if position_pts is None:
        position_pts = np.array([[0.0, 0.0, 1.0]] * (NmpcNode.N + 1))  # Example position setpoint
    else:
        position_pts = position_pts
    if attitude_pts is None:
        attitude_pts = np.array([[1.0, 0.0, 0.0, 0.0]] * (NmpcNode.N + 1))  # Example attitude setpoint
    else:
        attitude_pts = attitude_pts

    # Set parameters for the solver
    for j in range(NmpcNode.solver.N + 1):
        NmpcNode.p[14:21] = np.concatenate([position_pts[j], attitude_pts[j]], axis=0)
        NmpcNode.solver.set(j, 'p', NmpcNode.p)
        
def set_current_state(NmpcNode):
    
    """aggregates individual states to combined state of system
    """
    # , NmpcNode.state_timestamp
    NmpcNode.current_state = np.concatenate((NmpcNode.position, NmpcNode.attitude, NmpcNode.velocity, NmpcNode.angular_velocity, NmpcNode.thrust), axis=None)

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
    