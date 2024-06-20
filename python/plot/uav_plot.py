import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/riccardo/phd/research/nmpc_acados_py/uav_utils/python')
from utils import quaternion_to_euler

def plot(ref, path, q_path, u_path, dt, time_record):
    xref = ref[:,0]
    yref = ref[:,1]
    zref = ref[:,2]
    
    # Visualization
    path = np.array(path)
    plt.figure()
    plt.title('UAV Position in ENU frame')
    ax = plt.axes(projection='3d')
    ax.plot(yref, xref, -zref, c=[1,0,0], label='goal')
    ax.plot(path[:,1], path[:,0], -path[:,2])  # Swap x and y, negate z
    ax.axis('auto')
    ax.set_xlabel('y [m]')  # Swap x and y labels
    ax.set_ylabel('x [m]')
    ax.set_zlabel('-z [m]')  # Negate z label
    ax.legend()
    ax.grid(True)

    # Plot UAV attitude axes
    interval = 50
    for i in range(0, len(q_path), interval):
        euler_angles = quaternion_to_euler(q_path[i])
        origin = path[i]
        R = np.array([[math.cos(euler_angles[2])*math.cos(euler_angles[1]), math.cos(euler_angles[2])*math.sin(euler_angles[1])*math.sin(euler_angles[0])-math.sin(euler_angles[2])*math.cos(euler_angles[0]), math.cos(euler_angles[2])*math.sin(euler_angles[1])*math.cos(euler_angles[0])+math.sin(euler_angles[2])*math.sin(euler_angles[0])],
                  [math.sin(euler_angles[2])*math.cos(euler_angles[1]), math.sin(euler_angles[2])*math.sin(euler_angles[1])*math.sin(euler_angles[0])+math.cos(euler_angles[2])*math.cos(euler_angles[0]), math.sin(euler_angles[2])*math.sin(euler_angles[1])*math.cos(euler_angles[0])-math.cos(euler_angles[2])*math.sin(euler_angles[0])],
                  [-math.sin(euler_angles[1]), math.cos(euler_angles[1])*math.sin(euler_angles[0]), math.cos(euler_angles[1])*math.cos(euler_angles[0])]])
        x_axis = R @ np.array([1, 0, 0])
        y_axis = R @ np.array([0, 1, 0])
        z_axis = R @ np.array([0, 0, 1])
        ax.quiver(origin[1], origin[0], -origin[2], y_axis[1], y_axis[0], -y_axis[2], color='r', length=0.05)  # Swap x and y, negate z
        ax.quiver(origin[1], origin[0], -origin[2], x_axis[1], x_axis[0], -x_axis[2], color='g', length=0.05)  # Swap x and y, negate z
        ax.quiver(origin[1], origin[0], -origin[2], -z_axis[1], -z_axis[0], z_axis[2], color='b', length=0.05)  # Swap x and y, negate z
    ax.grid(True)

    plt.figure()
    plt.plot(time_record)
    plt.legend()
    plt.ylabel('CPU Time [s]')
    plt.grid(True)

    # Visualize inputs
    u_path = np.array(u_path)
    time = np.arange(0, len(u_path)*dt, dt)
    plt.figure()
    plt.suptitle("Rotor thrust - normalized")
    plt.subplot(2, 2, 1)
    plt.plot(time, u_path[:, 0])
    plt.ylabel('u1')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(time, u_path[:, 1])
    plt.ylabel('u2')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(time, u_path[:, 2])
    plt.ylabel('u3')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(time, u_path[:, 3])
    plt.ylabel('u4')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.tight_layout()
    
    # Visualize quaternion
    q_path = np.array(q_path)
    plt.figure()
    plt.suptitle("UAV attitude")
    plt.subplot(2, 2, 1)
    plt.plot(time, q_path[:, 0])
    plt.ylabel('qw')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(time, q_path[:, 1])
    plt.ylabel('qx')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(time, q_path[:, 2])
    plt.ylabel('qy')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(time, q_path[:, 3])
    plt.ylabel('qz')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()