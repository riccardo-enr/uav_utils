# UAV utils repo

## NMPC Utils

### Acceleration setpoint in NED frame to thrust and attitude reference

Thrust (in `acceleration2thrust_q` function):

$$
\begin{gather}
T_z = \sqrt{a_{NED,1}^2 + a_{NED,2}^2 + a_{NED,3}^2} \\
T = m \cdot T_z
\end{gather}
$$

```python
def acceleration_sp_to_thrust_q(NmpcNode, acc_sp_NED, yaw_sp):
    """
    Converts the desired acceleration in the NED frame to thrust and attitude quaternion.

    Args:
        NmpcNode: The NmpcNode object.
        acc_sp_NED: The desired acceleration in the NED frame, from NMPC.
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

    if acc_sp_NED.shape == (3, 1):
        acc_sp_NED = acc_sp_NED.reshape(3)
    elif acc_sp_NED.shape != (3,):
        raise ValueError("acc_sp_NED must be of shape (3,) or (3, 1), instead the shape is " + str(acc_sp_NED.shape) + ".")

    normal_acc = True
    max_acc_xy = max(np.abs(acc_sp_NED[0]), np.abs(acc_sp_NED[1]))
    if normal_acc:
        for i in range(len(acc_sp_NED) - 1):
            if np.abs(acc_sp_NED[i]) > 1.0:
                acc_sp_NED[i] = acc_sp_NED[i] / max_acc_xy

    Tz = np.sqrt(acc_sp_NED[0]**2 + acc_sp_NED[1]**2 + acc_sp_NED[2]**2)
    acceleration_sp_body = current_R @ acc_sp_NED

    # thrust = acceleration_sp_body[2] * NmpcNode.mass
    thrust = - Tz * NmpcNode.mass

    q_d, eul_d = acc2quaternion(NmpcNode, acc_sp_NED, yaw_sp, Tz, euler=True)
    thrust = normalize_thrust(thrust)
    thrust = np.clip(thrust, -1.0, 0.0)

    return thrust, q_d, eul_d
```


Attitude reference (in `acc2quaternion` function):

$$
\begin{gather}
\phi_{\text{des}} = \text{atan2}\left( \frac{-(u_t(1) \cdot \sin(\psi_{\text{ref}}) - u_t(2) \cdot \cos(\psi_{\text{ref}}))}{T_z}, \sqrt{1 - \left( \frac{-(u_t(1) \cdot \sin(\psi_{\text{ref}}) - u_t(2) \cdot \cos(\psi_{\text{ref}}))}{T_z} \right)^2} \right)
\\
\theta_{\text{des}} = \text{atan2}\left( -u_t(1) \cdot \cos(\psi_{\text{ref}}) - u_t(2) \cdot \sin(\psi_{\text{ref}}), M \cdot 9.81 - u_t(3) \right)
\end{gather}
$$

```python
def acc2quaternion(NmpcNode, acc_sp, psi, Tz, euler=False):
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

    num = - (acc_sp[0] * np.sin(psi) - acc_sp[1] * np.cos(psi)) / Tz
    phi_des = np.arctan2(num, np.sqrt(1 - num**2))
    theta_des = np.arctan2(
        - acc_sp[0] * np.cos(psi) - acc_sp[1] * np.sin(psi),
        NmpcNode.mass * g_ - acc_sp[2]
    )

    if euler:
        R_d = R.from_euler('xyz', [phi_des, theta_des, psi]).as_matrix()
        return rot2Quaternion(R_d), [phi_des, theta_des, psi]
    else:
        R_d = R.from_euler('xyz', [phi_des, theta_des, psi]).as_matrix()    
        return rot2Quaternion(R_d)
```