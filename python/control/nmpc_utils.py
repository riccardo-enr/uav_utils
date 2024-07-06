import numpy as np

def set_mpc_target_pos(NmpcNode):
    # Mock trajectory for testing
    position_pts = np.array([[0.0, 0.0, 1.0]] * (NmpcNode.N + 1))  # Example position setpoint
    attitude_pts = np.array([[1.0, 0.0, 0.0, 0.0]] * (NmpcNode.N + 1))  # Example attitude setpoint

    # Set parameters for the solver
    for j in range(NmpcNode.solver.N + 1):
        NmpcNode.p[14:21] = np.concatenate([position_pts[j], attitude_pts[j]], axis=0)
        NmpcNode.solver.set(j, 'p', NmpcNode.p)
        
def set_current_state(NmpcNode):
    
    """aggregates individual states to combined state of system
    """
    
    NmpcNode.current_state = np.concatenate((NmpcNode.position, NmpcNode.attitude, NmpcNode.velocity, NmpcNode.angular_velocity, NmpcNode.thrust, NmpcNode.state_timestamp), axis=None)

    #self.imu_data = np.concatenate((self.linear_accel_real, self.angular_accel_real, self.imu_timestamp), axis=None)
    
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

    return u_motor_values
    
    