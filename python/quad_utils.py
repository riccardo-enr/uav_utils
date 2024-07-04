import numpy as np
import casadi as cs

def ctrl_effectiveness_matrix(kt, l, config='x'):
    """
    Define the control effectiveness matrix for a quadrotor with an X configuration.

    Parameters:
    - kt: float or MX, the thrust factor
    - l: float or MX, the arm length
    - config: str, configuration type (default is 'x')

    Returns:
    - G: MX, the control effectiveness matrix
    """
    if config == '':
        config = 'x'

    if config == 'x':
        # Define symbolic variables for the arm lengths and thrust factor if they are not already symbolic
        # kt = cs.MX.sym('kt') if not isinstance(kt, cs.MX) else kt
        # l = cs.MX.sym('l') if not isinstance(l, cs.MX) else l

        lx1 = l * cs.cos(cs.pi / 4)
        ly1 = l * cs.sin(cs.pi / 4)
        lx2 = l * cs.cos(3 * cs.pi / 4)
        ly2 = l * cs.sin(3 * cs.pi / 4)
        lx3 = l * cs.cos(5 * cs.pi / 4)
        ly3 = l * cs.sin(5 * cs.pi / 4)
        lx4 = l * cs.cos(7 * cs.pi / 4)
        ly4 = l * cs.sin(7 * cs.pi / 4)

        G = cs.MX(4, 4)
        G[0, :] = [1, 1, 1, 1]
        G[1, :] = [ly1, ly2, ly3, ly4]
        G[2, :] = [-lx1, -lx2, -lx3, -lx4]
        G[3, :] = [-kt, -kt, kt, kt]

        return G