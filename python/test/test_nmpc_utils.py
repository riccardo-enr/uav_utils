import os
import sys
sys.stdout.write('test')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# dir_path = os.path.dirname(os.path.realpath(__file__))
# control_path = sys.path.append(os.path.join(dir_path, "../control"))
from control import nmpc_utils
# import control.nmpc_utils as nmpc_utils

class FakeNmpcNode():
    def __init__(self):
        self.mass = 2.0;
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])

def set_acc_sp():
    time = np.linspace(0, 5, num=int(5/0.1) + 1)
    x_sp = 0.25 * np.cos(time)
    # x_sp = np.full(len(time), 0.0)  # Create an array of 0.0 for the entire range of time
    y_sp = 0.5 * np.sin(time)
    # y_sp = np.full(len(time), 0.0)  # Create an array of 0.0 for the entire range of time
    z_sp = np.full(len(time), -12.0)  # Create an array of -12.0 for the entire range of time
    yaw_sp = 0.0

    # Initialize acc_sp with the correct shape
    acc_sp = np.zeros((len(time), 3))

    for i in range(len(time)):
        acc_sp[i] = [x_sp[i], y_sp[i], z_sp[i]]

    return acc_sp, time

def main():
    fake_uav = FakeNmpcNode()
    try:
        acc_sp, time = set_acc_sp()
        thrust = np.zeros(len(acc_sp))
        q_d = np.zeros((len(acc_sp), 4))
        eul_d = np.zeros((len(acc_sp), 3))
        print(len(acc_sp))
        for i in range(len(acc_sp)):
            print("acc_sp: ", acc_sp[i])
            thrust[i], q_d[i], eul_d[i] = nmpc_utils.acceleration_sp_to_thrust_q(fake_uav, acc_sp[i], 0.0)
            print("Thrust: ", thrust[i])
            print("q_d: ", q_d[i])
            print("eul_d: ", eul_d[i])

        # Plotting the acceleration components
        x_sp = acc_sp[:, 0]
        y_sp = acc_sp[:, 1]
        z_sp = acc_sp[:, 2]

        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8))

        axs1[0].plot(time, x_sp, label='x_sp')
        axs1[0].set_xlabel('Time [s]')
        axs1[0].set_ylabel('x_sp')
        axs1[0].legend()

        axs1[1].plot(time, y_sp, label='y_sp')
        axs1[1].set_xlabel('Time [s]')
        axs1[1].set_ylabel('y_sp')
        axs1[1].legend()

        axs1[2].plot(time, z_sp, label='z_sp')
        axs1[2].set_xlabel('Time [s]')
        axs1[2].set_ylabel('z_sp')
        axs1[2].legend()

        fig1.tight_layout()

        # Plotting thrust and quaternion components in a new figure
        fig2, axs2 = plt.subplots(5, 1, figsize=(10, 12))

        axs2[0].plot(time, thrust, label='Thrust')
        axs2[0].set_xlabel('Time [s]')
        axs2[0].set_ylabel('Thrust')
        axs2[0].legend()
        axs2[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs2[1].plot(time, q_d[:, 0], label='q_d[0]')
        axs2[1].set_xlabel('Time [s]')
        axs2[1].set_ylabel('q_d[0]')
        axs2[1].legend()
        axs2[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs2[2].plot(time, q_d[:, 1], label='q_d[1]')
        axs2[2].set_xlabel('Time [s]')
        axs2[2].set_ylabel('q_d[1]')
        axs2[2].legend()
        axs2[2].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs2[3].plot(time, q_d[:, 2], label='q_d[2]')
        axs2[3].set_xlabel('Time [s]')
        axs2[3].set_ylabel('q_d[2]')
        axs2[3].legend()
        axs2[3].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs2[4].plot(time, q_d[:, 3], label='q_d[3]')
        axs2[4].set_xlabel('Time [s]')
        axs2[4].set_ylabel('q_d[3]')
        axs2[4].legend()
        axs2[4].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        fig2.tight_layout()

        # Plotting Euler angles in a third figure
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8))

        axs3[0].plot(time, eul_d[:, 0], label='eul_d[0]')
        axs3[0].set_xlabel('Time [s]')
        axs3[0].set_ylabel('eul_d[0]')
        axs3[0].legend()
        axs3[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs3[1].plot(time, eul_d[:, 1], label='eul_d[1]')
        axs3[1].set_xlabel('Time [s]')
        axs3[1].set_ylabel('eul_d[1]')
        axs3[1].legend()
        axs3[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        axs3[2].plot(time, eul_d[:, 2], label='eul_d[2]')
        axs3[2].set_xlabel('Time [s]')
        axs3[2].set_ylabel('eul_d[2]')
        axs3[2].legend()
        axs3[2].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        fig3.tight_layout()

        # Display all figures
        plt.show()
    except KeyboardInterrupt:
        plt.close('all')
        print("Plotting was closed.")

if __name__ == "__main__":
    main()
