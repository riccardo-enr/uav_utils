import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/riccardo/phd/research/nmpc_acados_py/uav_utils/python')
from utils import quaternion_to_euler

def plot(ref, path, q_path, u_path, dt, time_record):
    plot_position(ref, path)
    plot3d(ref, path, q_path)
    # plot_attitude(path, q_path)
    plot_cpu_time(time_record)
    plot_inputs(u_path, dt)
    plot_quaternion(q_path)
    plt.show()

def plot_position(ref, path):
    xref = ref[:, 0]
    yref = ref[:, 1]
    zref = ref[:, 2]

    # Convert path to numpy array for easier manipulation
    path = np.array(path)
    x_path = path[:, 0]
    y_path = path[:, 1]
    z_path = path[:, 2]

    # Time vector for x-axis
    time = np.arange(len(xref))

    # Create subplots
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig, axs = plt.subplots(3, 1)

    # Plot X
    axs[0].plot(time, xref, label='X Reference', color='r', linestyle='--')
    axs[0].plot(time, x_path, label='X Path', color='b')
    axs[0].set_title('X Position Over Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X Position')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Y
    axs[1].plot(time, yref, label='Y Reference', color='r', linestyle='--')
    axs[1].plot(time, y_path, label='Y Path', color='b')
    axs[1].set_title('Y Position Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Y Position')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Z
    axs[2].plot(time, zref, label='Z Reference', color='r', linestyle='--')
    axs[2].plot(time, z_path, label='Z Path', color='b')
    axs[2].set_title('Z Position Over Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Z Position')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    
def plot3d(ref, path, q_path):
    xref = ref[:,0]
    yref = ref[:,1]
    zref = ref[:,2]
    
    # Visualization
    path = np.array(path)
    q_path = np.array(q_path)
    
    plt.figure()
    plt.title('UAV Position in ENU frame')
    ax = plt.axes(projection='3d')
    ax.plot(yref, xref, -zref, c=[1,0,0], label='goal')
    ax.plot(path[:,1], path[:,0], - path[:,2], label='path')  # Swap x and y, negate z
    ax.axis('auto')
    ax.set_xlabel('y [m]')  # Swap x and y labels
    ax.set_ylabel('x [m]')
    ax.set_zlabel('-z [m]')  # Negate z label
    ax.legend()
    ax.grid(True)

    # Define time variable
    time = np.arange(0, len(q_path), 1)
    
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

def plot_cpu_time(time_record):
    plt.figure()
    plt.plot(time_record)
    plt.legend()
    plt.ylabel('CPU Time [s]')
    plt.grid(True)

def plot_inputs(u_path, dt):
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

def plot_quaternion(q_path):
    # Define time variable
    time = np.arange(0, len(q_path), 1)
    
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