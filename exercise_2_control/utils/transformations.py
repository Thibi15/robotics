
import casadi as ca
import numpy as np


def rotate_x_SE3(angle):
    """Homogenous transform matrix for a rotation around X"""

    if isinstance(angle, ca.SX):
        T = ca.SX.eye(4)
    elif isinstance(angle, ca.MX):
        T = ca.MX.eye(4)
    else:
        T = np.eye(4)

    T[1, 1] = ca.cos(angle)
    T[1, 2] = -ca.sin(angle)
    T[2, 1] = ca.sin(angle)
    T[2, 2] = ca.cos(angle)

    return T


def inverse_SE3(T):
    """
    Calculate the inverse of a transformation matrix
    """

    # extract the rotation matrix and the position vector
    R = T[:3, :3]
    p = T[:3, 3]

    if isinstance(T, ca.SX):
        T_inv = ca.MX.eye(4)
    else:
        T_inv = ca.MX.eye(4)

    
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = - R.T @ p

    return T_inv


def rotate_y_SE3(angle):
    """Homogenous transform matrix for a rotation around Y"""

    if isinstance(angle, ca.SX):
        T = ca.SX.eye(4)
    elif isinstance(angle, ca.MX):
        T = ca.MX.eye(4)
    else:
        T = np.eye(4)

    T[0, 0] = ca.cos(angle)
    T[0, 2] = ca.sin(angle)
    T[2, 0] = -ca.sin(angle)
    T[2, 2] = ca.cos(angle)

    return T


def rotate_z_SE3(angle):
    """Homogenous transform matrix for a rotation around Z"""

    if isinstance(angle, ca.SX):
        T = ca.SX.eye(4)
    elif isinstance(angle, ca.MX):
        T = ca.MX.eye(4)
    else:
        T = np.eye(4)

    T[0, 0] = ca.cos(angle)
    T[0, 1] = -ca.sin(angle)
    T[1, 0] = ca.sin(angle)
    T[1, 1] = ca.cos(angle)

    return T


def translate_SE3(x, y, z):
    """Homogenous transform matrix for translation"""
    if isinstance(x, ca.SX) or isinstance(y, ca.SX) or isinstance(z, ca.SX):
        T = ca.SX.eye(4)
    elif isinstance(x, ca.MX) or isinstance(y, ca.MX) or isinstance(z, ca.MX):
        T = ca.MX.eye(4)
    else:
        T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    return T

def crossvec_SO3(M):
    '''
    Extracts a vector from a skew-symetric matrix
    '''
    # return ca.vertcat(M[2, 1]-M[1, 2], M[0, 2]-M[2, 0], M[1, 0] - M[0, 1]) / 2.0
    return ca.vertcat(M[2, 1], M[0, 2], M[1, 0])


def logm_SO3(R):
    '''
    Calculates the matrix logarithm of a rotation matrix
    '''

    # function based on the Erwin's frames.py, but with casadi
    axis = crossvec_SO3(R)
    s_a = ca.norm_2(axis)
    c_a = (ca.trace(R)-1)/2.0

    c_a = ca.if_else(c_a < -1, -1, c_a)
    c_a = ca.if_else(c_a > 1,  1, c_a)
    alpha = ca.if_else(s_a < 1E-17,  1/2.0, ca.atan2(s_a, c_a)/s_a/2.0)

    w_hat = (R-R.T)*alpha
    return w_hat


def posvec_SE3(T):
    '''
    Selects the position of a transformation matrix
    '''

    return T[:3, 3]


def logm_SE3(T: ca.MX):
    '''
    Calculates the log of SE3 (transformation matrix) as a casadi expression

    Parameters
    ----------
    T : Transformation matrix as a casadi expression (4x4)

    Returns
    -------
    log(T) as a casadi expression (4x4)
    '''

    # function based on the Erwin's frames.py, but with casadi

    R = T[:3, :3]
    p = T[:3, 3]

    # calculate the rotation angle and axis
    omega_hat = logm_SO3(R)
    omega = crossvec_SO3(omega_hat)
    theta = ca.norm_2(omega)

    G = (ca.MX.eye(3)-R) @ omega_hat/theta + (omega @ omega.T)/theta

    v = ca.inv(G) @ p * theta

    v = ca.if_else(theta == 0, p, v)

    result = ca.MX.zeros(4, 4)
    result[:3, :3] = omega_hat
    result[:3, 3] = v

    return result


def SE3_from_RPY(roll: ca.MX, pitch: ca.MX, yaw: ca.MX):
    '''
    Constructs a transformation matrix from roll, pitch and yaw

    Parameters
    ----------
    roll: roll angle
    pitch: pitch angle
    yaw: yaw angle


    Returns
    -------
    4x4 transformation matrix
    '''
    return rotate_z_SE3(yaw)@rotate_y_SE3(pitch)@rotate_x_SE3(roll)



def SE3_from_xyz_rpy(xyz, rpy):
    """Gives a symbolic transformation matrix."""
    T_translate = translate_SE3(xyz[0], xyz[1], xyz[2])
    T_rotate = SE3_from_RPY(rpy[0], rpy[1], rpy[2])
    # T[0, 3] = xyz[0]
    # T[1, 3] = xyz[1]
    # T[2, 3] = xyz[2]
    return T_translate @ T_rotate


def wedge(S):
    """
    Casadi helper functions
    S is a skew-symmetric matrix
    s is the 3-vector extracted from S
    This code was taken from https://github.com/nmansard/jnrh2023
    """
    s = ca.vertcat(S[2, 1], S[0, 2], S[1, 0])
    return s


def logm_SO3_approx(R):
    """
    R is a rotation matrix not far from the identity
    w is the approximated rotation vector equivalent to R 

    This code was taken from https://github.com/nmansard/jnrh2023

    Parameters
    ----------
    R: 3x3 rotation matrix

    Returns
    -------
    w: 3x1 vector
      Approximation of the log of a Rotation matrix
    """
    w = wedge(R - R.T) / 2
    return w


def axis_angle_from_rotation_matrix(R):

    return crossvec_SO3(logm_SO3(R))
