import numpy as np

def is_setpoint_reached(setpoint, current_position, threshold_pos, threshold_att) -> bool:
    if len(setpoint) == 3:
        # Assume only position is to be checked
        return np.linalg.norm(np.array(setpoint) - np.array(current_position)) < threshold_pos
    elif len(setpoint) == 4:
        # Assume position and yaw are to be checked
        if len(current_position) == 4:
            return np.linalg.norm(np.array(setpoint[:3]) - np.array(current_position[:3])) < threshold_pos and np.abs(setpoint[3] - current_position[3]) < threshold_att
        elif len(current_position) == 7:
            # assume current attitude is in quaternion form
            