from pinocchio import casadi as cpin
import casadi as ca
import numpy as np
import copy


def integrate_RK4(x_expr, u_expr, xdot_expr, dt, N_steps=1):
    h = dt / N_steps
    x_end = x_expr
    xdot_fun = ca.Function('xdot', [x_expr, u_expr], [xdot_expr])

    for _ in range(N_steps):
        k_1 = xdot_fun(x_end, u_expr)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr)
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr)
        k_4 = xdot_fun(x_end + k_3 * h, u_expr)

        x_end = x_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    F_expr = x_end

    return F_expr


class TorqueRobotSim:

    def __init__(self, robot, add_model_mismatch=False, alpha_internal=50):
        self.robot = copy.deepcopy(robot)

        if add_model_mismatch:
            self.update_mass()
        self.model_missmatch = add_model_mismatch
        self.alpha_internal = alpha_internal

        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()

        # Define casadi symbols for the joints
        cq = ca.SX.sym('q', self.robot.model.nq)
        cdq = ca.SX.sym('dq', self.robot.model.nq)
        ctau = ca.SX.sym('tau', self.robot.model.nq)

        # Compute the forward dynamics
        cddq = cpin.aba(self.cmodel, self.cdata,
                             cq, cdq, ctau)

        x_expr = ca.vertcat(cq, cdq)
        u_expr = ca.vertcat(ctau)
        xdot_expr = ca.vertcat(cdq, cddq)

        cdt = ca.SX.sym("dt")

        x_next_sym = integrate_RK4(x_expr=x_expr, u_expr=u_expr,
                                        xdot_expr=xdot_expr, dt=cdt, N_steps=2)

        self.f_state_transition = ca.Function(
            "f_state_transition", [x_expr, u_expr, cdt], [x_next_sym], ["x0", "u0", "dt"], ["x1"])
        
        self.u_prev = 0

        self.reset()

    def update_mass(self):
        """
        Slightly increase the mass of each of the links.
        Used for creating a model mismatch.
        """
        for i in range(len(self.robot.model.inertias)):
            inertia = self.robot.model.inertias[i]
            inertia.mass += 0.01
            self.robot.model.inertias[i] = inertia

    def reset(self):
        """
        Resets the controller:
        - starts at initial position
        - clears the logs used for plotting
        """
        self.x0 = np.append(self.robot.q0, np.zeros_like(self.robot.q0))

        self.q_list = [self.x0[:self.robot.nq]]
        self.dq_list = [self.x0[self.robot.nq:]]
        self.u_list = []

        self.t_list = [0]

    def step(self, u, dt):
        """
        Updates the state of the robot simulator by applying the control input over a time step.
        Args:
            u (numpy.ndarray): Control input (joint torque) vector.
            dt (float): Time step duration.
        Returns:
            None
        """
        if self.model_missmatch:
            # Clip the control input to stay within the effort limits
            u = np.clip(u, -self.robot.model.effortLimit*0.3, self.robot.model.effortLimit*0.3)

            # add bandwidth limitation
            u = self.u_prev - dt*self.alpha_internal*(self.u_prev - u)
            self.u_prev = u
        
        self.x0 = self.f_state_transition(self.x0, u, dt).full().flatten()

        self.q_list.append(self.x0[:self.robot.nq])
        self.dq_list.append(self.x0[self.robot.nq:])
        self.u_list.append(u)
        self.t_list.append(self.t_list[-1] + dt)
        

    def get_joint_state(self):
        q = self.x0[:self.robot.nq]
        dq = self.x0[self.robot.nq:]
        return q, dq

    def get_joint_pos(self):
        q = self.x0[:self.robot.nq]
        return q


class VelocityRobotSim:

    def __init__(self, q0, n_joints, alpha_internal = 20):

        self.n_joints = n_joints
        # Define casadi symbols for the joints
        cq = ca.SX.sym('q', self.n_joints)
        cdq = ca.SX.sym('dq', self.n_joints)
        cdq_ref = ca.SX.sym('cdq_ref', self.n_joints)

        self.q0 = q0

        x_expr = ca.vertcat(cq, cdq)
        u_expr = ca.vertcat(cdq_ref)
        xdot_expr = ca.vertcat(cdq, alpha_internal*(cdq_ref - cdq))

        cdt = ca.SX.sym("dt")

        x_next_sym = integrate_RK4(x_expr=x_expr, u_expr=u_expr,
                                   xdot_expr=xdot_expr, dt=cdt, N_steps=2)

        self.f_state_transition = ca.Function(
            "f_state_transition", [x_expr, u_expr, cdt], [x_next_sym], ["x0", "u0", "dt"], ["x1"])

        self.reset()

    def reset(self):
        """
        Resets the controller:
        - starts at initial position
        - clears the logs used for plotting
        """
        self.x0 = np.append(self.q0, np.zeros_like(self.q0))

        self.q_list = [self.x0[:self.n_joints]]
        self.dq_list = [self.x0[self.n_joints:]]
        self.u_list = []

        self.t_list = [0]

    def step(self, u, dt):
        """
        Updates the state of the robot simulator by applying the control input over a time step.
        Args:
            u (numpy.ndarray): Control input (joint torque) vector.
            dt (float): Time step duration.
        Returns:
            None
        """

        self.x0 = self.f_state_transition(self.x0, u, dt).full().flatten()

        self.q_list.append(self.x0[:self.n_joints])
        self.dq_list.append(self.x0[self.n_joints:])
        self.u_list.append(u)
        self.t_list.append(self.t_list[-1] + dt)

    def get_joint_state(self):
        q = self.x0[:self.n_joints]
        dq = self.x0[self.n_joints:]
        return q, dq

    def get_joint_pos(self):
        q = self.x0[:self.n_joints]
        return q


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initial joint positions
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Number of joints
    n_joints = len(q0)

    # Create a VelocityRobotSim instance
    velocity_robot = VelocityRobotSim(q0, n_joints)

    # Simulation parameters
    dt = 0.01  # time step
    total_time = 0.5  # total simulation time
    steps = int(total_time / dt)

    # Control input (step response)
    u = np.array([0.1] * n_joints)

    # Simulate
    for _ in range(steps):
        velocity_robot.step(u, dt)

    # Plot results
    t = velocity_robot.t_list
    q_list = np.array(velocity_robot.q_list)
    dq_list = np.array(velocity_robot.dq_list)

    plt.figure()
    for i in range(n_joints):
        plt.subplot(n_joints, 1, i + 1)
        # plt.plot(t, q_list[:, i], label=f'Joint {i + 1} Position')
        plt.plot(t, dq_list[:, i], label=f'Joint {i + 1} Velocity')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.show()