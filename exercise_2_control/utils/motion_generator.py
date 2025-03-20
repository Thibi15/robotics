import rockit
import numpy as np


class MotionPlanner:
    """
    A basic motion planner using rockit
    """

    def __init__(self, q_start, q_goal, dq_max, ddq_max):
        """
        Initialize the optimal control problem for a robotic system.

        Parameters:
            q_start (numpy.ndarray): The initial state of the system.
            q_goal (numpy.ndarray): The desired goal state of the system.
            ddq_max (float): The maximum allowable acceleration.
        """

        self.q_start = q_start
        self.q_goal = q_goal

        # formulate an optimal control problem
        ocp = rockit.Ocp(T=rockit.FreeTime(1))

        self.num_joints = q_start.shape[0]

        # define the system dynamics
        p = ocp.state(self.num_joints)
        v = ocp.state(self.num_joints)
        u = ocp.control(self.num_joints)

        # define the system dynamics
        ocp.set_der(p, v)
        ocp.set_der(v, u)

        # specify the objective
        ocp.add_objective(ocp.T)
        ocp.add_objective(ocp.integral(1e-5*u.T@u))


        # constrain the input
        ocp.subject_to(-ddq_max <= (u <= ddq_max))
        ocp.subject_to(-dq_max <= (v <= dq_max))


        # constrain start and end state
        ocp.subject_to(ocp.at_t0(p) == q_start)
        ocp.subject_to(ocp.at_t0(v) == 0)

        ocp.subject_to(ocp.at_tf(p) == q_goal)
        ocp.subject_to(ocp.at_tf(v) == 0)

        ocp.method(rockit.MultipleShooting(N=40))

        # specify the solver options
        options = {"expand": True, 'print_time': False,
                   'ipopt.print_level': 0, 'ipopt.sb': 'yes'}
        ocp.solver("ipopt", options)

        sol = ocp.solve()

        self.total_time = sol.value(ocp.T)
        self.position_sampler = sol.sampler(p)
        self.velocity_sampler = sol.sampler(v)
        self.acceleration_sampler = sol.sampler(u)

    def __call__(self, t):
        """
        Get the position, velocity, and acceleration at time t.

        Parameters:
            t (float or list or numpy.ndarray): The time at which to sample the motion.

        Returns:
            tuple: A tuple containing the position, velocity, and acceleration at time t.
        """
        if isinstance(t, (list, np.ndarray)):
            positions = []
            velocities = []
            accelerations = []
            for time in t:
                pos, vel, accel = self._sample_at_time(time)
                positions.append(pos)
                velocities.append(vel)
                accelerations.append(accel)
            return np.array(positions), np.array(velocities), np.array(accelerations)
        else:
            return self._sample_at_time(t)

    def _sample_at_time(self, t):
        """
        Helper function to sample the motion at a specific time.

        Parameters:
            t (float): The time at which to sample the motion.

        Returns:
            tuple: A tuple containing the position, velocity, and acceleration at time t.
        """
        if t < 0:
            pos = self.q_start
            vel = np.zeros(self.num_joints)
            accel = np.zeros(self.num_joints)
        elif t > self.total_time:
            pos = self.q_goal
            vel = np.zeros(self.num_joints)
            accel = np.zeros(self.num_joints)
        else:
            pos = self.position_sampler(t)
            vel = self.velocity_sampler(t)
            accel = self.acceleration_sampler(t)

        return pos, vel, accel


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define start and goal states
    q_start = np.array([0, -1])
    q_goal = np.array([1, 1])
    ddq_max = 1.0

    # Create a motion planner instance
    planner = MotionPlanner(q_start, q_goal, ddq_max)

    # Sample the motion at different time instances
    times = np.linspace(0, planner.total_time, 100)
    positions = []
    velocities = []
    accelerations = []

    for t in times:
        pos, vel, accel = planner(t)
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(accel)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    # Plot the results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(times, positions)
    plt.title('Position')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')

    plt.subplot(3, 1, 2)
    plt.plot(times, velocities)
    plt.title('Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')

    plt.subplot(3, 1, 3)
    plt.plot(times, accelerations)
    plt.title('Acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration')

    plt.tight_layout()
    plt.show()