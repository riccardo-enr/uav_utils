import os
import numpy as np
import control.nmpc_utils as nmpc_utils

def set_acc_sp():
    time = np.linspace(0, 5, num=int(5/0.1) + 1)
    x_sp = 0.25 * np.cos(time)
    y_sp = 0.25 * np.sin(time)
    z_sp = - 23.0 * time
    yaw_sp = 0.0
    return np.array([x_sp, y_sp, z_sp])

def main():
    acc_sp = set_acc_sp()
    thrust = np.zeros(len(acc_sp))
    q_d = np.zeros((len(acc_sp), 4))
    for i in range(len(acc_sp)):
       print("acc_sp: ", acc_sp[i])
       thrust[i], q_d[i] = nmpc_utils.acceleration_sp_to_thrust_q(None, acc_sp[i], 0.0)
       print("Thrust: ", thrust[i])
       print("q_d: ", q_d[i])

if __name__ == "__main__":
    main()
