#!/usr/bin/env python3
import os
import time
import numpy as np
import scipy
import functions as functions

from casadi import SX, vertcat, horzcat, inv, cross, sum1, sin, cos
# WARNING: imports outside of constants will not trigger a rebuild
# from openpilot.selfdrive.modeld.constants import ModelConstants

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver # Forse togliere

if __name__ == '__main__':  # generating code
  from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
# else:
#   from nmpc_ros2_py.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

NMPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.realpath(os.path.join(NMPC_DIR, "./../c_generated_code"))
ROS_WS_INSTALL_DIR = os.path.realpath(os.path.join(NMPC_DIR, "./../../../install"))
EXPORT_DIR = os.path.realpath(os.path.join(ROS_WS_INSTALL_DIR, "./lib/nmpc_ros2_py/c_generated_code"))
JSON_FILE = os.path.realpath(os.path.join(ROS_WS_INSTALL_DIR, "./lib/nmpc_ros2_py/acados_ocp_quad.json"))

X_DIM = 13
U_DIM = 4
P_DIM = 17 # TODO: Da cambiare
COST_E_DIM = 7
COST_DIM = COST_E_DIM + U_DIM
# SPEED_OFFSET = 10.0
MODEL_NAME = 'quadrotor'
ACADOS_SOLVER_TYPE = 'SQP_RTI'
N = 20

def error_function(x, y_ref):
    """Error function for MPC
        difference of position and attitude from reference
        use sub-function for calculating quaternion error

    Args:
        x (casadi SX): current state of position and attitude
        y_ref (casadi SX): desired reference

    Returns:
        casadi SX: vector containing position and attiude error (attitude error only regarding yaw)
    """
    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    
    
    p_err = x[0:3] - p_ref
    q_err = functions.quaternion_error_casadi(x[3:7], q_ref)[2]
    
    return vertcat(p_err, q_err)

def gen_quad_model() -> AcadosModel:

    model_name = "quad_ode"

    # Define parameters
    m = SX.sym("m")  # Mass of the quadrotor
    g = SX.sym("g")  # Acceleration due to gravity
    jxx = SX.sym("jxx")  # diagonal components of inertia matrix
    jyy = SX.sym("jyy")
    jzz = SX.sym("jzz")

    d_x0 = SX.sym("d_x0")  # distances of motors from respective axis
    d_x1 = SX.sym("d_x1")
    d_x2 = SX.sym("d_x2")
    d_x3 = SX.sym("d_x3")
    d_y0 = SX.sym("d_y0")
    d_y1 = SX.sym("d_y1")
    d_y2 = SX.sym("d_y2")
    d_y3 = SX.sym("d_y3")
    c_tau = SX.sym("c_tau")  # rotor drag torque constant
    p_ref = SX.sym("p_ref", 3)  # reference variables for setpoint
    q_ref = SX.sym("q_ref", 4)
    lin_acc_offset = SX.sym("lin_acc_offset", 3)
    ang_acc_offset = SX.sym("ang_acc_offset", 3)
    
    # combine parameters to single vector
    params = vertcat(
        m,
        g,
        jxx,
        jyy,
        jzz,
        d_x0,
        d_x1,
        d_x2,
        d_x3,
        d_y0,
        d_y1,
        d_y2,
        d_y3,
        c_tau,
        p_ref,
        q_ref,
        lin_acc_offset,
        ang_acc_offset
    )

    # Define state variables
    p_WB = SX.sym("p_WB", 3)  # Position of the quadrotor (x, y, z)
    q_WB = SX.sym(
        "q_WB", 4
    )  # Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB = SX.sym("v_WB", 3)  # Linear velocity of the quadrotor
    omega_B = SX.sym("omega_B", 3)  # Angular velocity of the quadrotor in body frame
    thrust = SX.sym("T", 4)
    x = vertcat(p_WB, q_WB, v_WB, omega_B, thrust)

    # Define control inputs
    thrust_set = SX.sym("T_set", 4)  # Thrust produced by the rotors

    # Inertia matrix
    J = vertcat(horzcat(jxx, 0, 0),
                horzcat(0, jyy, 0),
                horzcat(0, 0, jzz))

    # thrust allocation matrix
    P = vertcat(
        horzcat(-d_x0, +d_x1, +d_x2, -d_x3),
        horzcat(-d_y0, +d_y1, -d_y2, +d_y3),
        horzcat(-c_tau, c_tau, -c_tau, c_tau),
    )

    # xdot
    p_WB_dot = SX.sym("p_WB_dot", 3)  # derivative of Position of the quadrotor (x, y, z)
    q_WB_dot = SX.sym("q_WB_dot", 4)  # derivative of Orientation of the quadrotor as a unit quaternion (qw, qx, qy, qz)
    v_WB_dot = SX.sym("v_WB_dot", 3)  # derivative of Linear velocity of the quadrotor
    omega_B_dot = SX.sym("omega_B_dot", 3)  # derivative of Angular velocity of the quadrotor in body frame
    thrust_dot = SX.sym("T_dot", 4)   # derivative of thrust

    xdot = vertcat(p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, thrust_dot)

    f_expl = vertcat(
        v_WB,
        functions.quat_derivative_casadi(q_WB, omega_B),
        functions.quat_rotation_casadi(vertcat(0, 0, sum1(thrust)), q_WB) / m + vertcat(0, 0, g) + lin_acc_offset,
        inv(J) @ ((P @ thrust - cross(omega_B, J @ omega_B))) + ang_acc_offset,
        (thrust_set - thrust) * 25,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = thrust_set
    model.p = params
    model.name = model_name

    return model

class QuadOCP:
    def __init__(self):
        # variables for ACADOS MPC
        self.N_horizon = 20
        self.Tf = 1
        self.nx = 17
        self.nu = 4
        self.Tmax = 1
        self.Tmin = 0
        self.vmax = 3
        self.angular_vmax = 1.5
        self.max_angle_q = 1
        self.max_motor_rpm = 1000
        
        # parameters for system model
        self.m = 2.0
        
        self.g = -9.81
        self.jxx = 0.08612
        self.jyy = 0.08962
        self.jzz = 0.16088
       
        l = 0.25
        
        self.d_x0 = np.cos(np.pi/4) * l
        self.d_x1 = np.cos(np.pi/4) * l
        self.d_x2 = np.cos(np.pi/4) * l
        self.d_x3 = np.cos(np.pi/4) * l
        self.d_y0 = np.sin(np.pi/4) * l
        self.d_y1 = np.sin(np.pi/4) * l
        self.d_y2 = np.sin(np.pi/4) * l
        self.d_y3 = np.sin(np.pi/4) * l
        self.c_tau = 0.016
        self.hover_thrust = -self.g*self.m/4
        
        self.params = np.asarray([self.m,
                            self.g,
                            self.jxx,
                            self.jyy,
                            self.jzz,
                            self.d_x0,
                            self.d_x1,
                            self.d_x2, 
                            self.d_x3, 
                            self.d_y0,
                            self.d_y1,
                            self.d_y2,
                            self.d_y3,
                            self.c_tau])
        
        #setpoint variables
        self.position_setpoint = np.array([0,0,2])
        self.velocity_setpoint = np.zeros(3)
        self.attitude_setpoint = np.asarray([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
        self.roll_setpoint = 0
        self.pitch_setpoint = 0
        self.yaw_setpoint = 0
        self.angular_velocity_setpoint = np.zeros(3)
        self.setpoint = np.concatenate((self.position_setpoint, self.attitude_setpoint), axis=None)
        
        self.parameters = np.concatenate((self.params, self.setpoint, np.zeros(6)), axis=None)

    def gen_ocp(self):
        ocp = AcadosOcp()
        ocp.model = gen_quad_model()

        ocp.dims.N = self.N_horizon
        ocp.parameter_values = self.parameters

        # define weighing matrices
        Q_p = np.diag([40,40,200])*5
        Q_q = np.eye(1)*100
        Q_mat = scipy.linalg.block_diag(Q_p, Q_q)

        R_U = np.eye(4)
        
        Q_p_final = np.diag([28,28,200])*30
        Q_q_final = np.eye(1)*100
        Q_mat_final = scipy.linalg.block_diag(Q_p_final, Q_q_final)
        
        # set cost module
        x = ocp.model.x[0:7]
        u = ocp.model.u
        
        ref = ocp.model.p[14:21]
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        
        ocp.model.cost_expr_ext_cost = error_function(x, ref).T @ Q_mat @ error_function(x,ref) + u.T @ R_U @ u 
        ocp.model.cost_expr_ext_cost_e = error_function(x, ref).T @ Q_mat_final @ error_function(x, ref)
        
        # set constraints
        Tmin = self.Tmin
        Tmax = self.Tmax
        vmax = self.vmax
        angular_vmax = self.angular_vmax
        
        # input constraints        
        ocp.constraints.lbu = np.array([Tmin, Tmin, Tmin, Tmin])
        ocp.constraints.ubu = np.array([Tmax, Tmax, Tmax, Tmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
            
        # state constraints     
        ocp.constraints.lbx = np.array([-self.max_angle_q, -self.max_angle_q, -vmax, -vmax, -vmax, -angular_vmax, -angular_vmax, -angular_vmax])
        ocp.constraints.ubx = np.array([+self.max_angle_q, +self.max_angle_q, +vmax, +vmax, +vmax, +angular_vmax, +angular_vmax, +angular_vmax])
        ocp.constraints.idxbx = np.array([4, 5, 7, 8, 9, 10, 11, 12])
        
        # set initial state
        # ocp.constraints.x0 = self.current_state[:-1]
        ocp.constraints.x0 = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        # set prediction horizon
        ocp.solver_options.qp_solver_cond_N = self.N_horizon
        ocp.solver_options.tf = self.Tf

        # set solver options
        ocp.solver_options.levenberg_marquardt = 10.0
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'EXACT'

        ocp.code_export_directory = EXPORT_DIR
        return ocp

class Nmpc:
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.zeros(X_DIM)
        # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
        ocp = QuadOCP().gen_ocp()
        self.solver = AcadosOcpSolver(ocp, json_file=JSON_FILE, build=False, generate=False)
        self.reset(x0)

    def reset(self, x0=None):
        if x0 is None:
            x0 = np.zeros(X_DIM)
        self.x_sol = np.zeros((N+1, X_DIM))
        self.u_sol = np.zeros((N, 1))
        self.yref = np.zeros((N+1, COST_DIM))
        for i in range(N):
            self.solver.cost_set(i, "yref", self.yref[i])
        self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

        # Somehow needed for stable init
        for i in range(N+1):
            self.solver.set(i, 'x', np.zeros(X_DIM))
            self.solver.set(i, 'p', np.zeros(P_DIM))
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        self.solver.solve()
        self.solution_status = 0
        self.solve_time = 0.0
        self.cost = 0

        # def set_weights # Non server -> Già settati in gen_ocp 

    def run(self, x0, p, position_pts, attitude_pts):
        """
        Run the NMPC solver with the provided initial state, parameters, and setpoints.

        Args:
            x0 (numpy.ndarray): Initial state of the system.
            p (numpy.ndarray): Parameters for the solver.
            position_pts (numpy.ndarray): Reference position points.
            attitude_pts (numpy.ndarray): Reference attitude points.
        """
        x0_cp = np.copy(x0)
        p_cp = np.copy(p)
        
        # Set the initial state constraints
        self.solver.constraints_set(0, "lbx", x0_cp)
        self.solver.constraints_set(0, "ubx", x0_cp)
        
        # Update yref with position and attitude references
        self.yref[:, 0:3] = position_pts
        self.yref[:, 3:7] = attitude_pts

        # Set references for each time step
        for i in range(N):
            self.solver.cost_set(i, "yref", self.yref[i])
            self.solver.set(i, "p", p_cp)
        
        # Set the reference for the last step (end cost)
        self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

        # Measure the time taken to solve
        t = time.monotonic()
        self.solution_status = self.solver.solve()
        self.solve_time = time.monotonic() - t

        # Retrieve the solution
        for i in range(N+1):
            self.x_sol[i] = self.solver.get(i, 'x')
        for i in range(N):
            self.u_sol[i] = self.solver.get(i, 'u')
        
        self.cost = self.solver.get_cost()

if __name__ == "__main__":
    quad_ocp = QuadOCP()
    ocp = quad_ocp.gen_ocp()
    
    AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
    # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    AcadosOcpSolver.build(ocp.code_export_directory)