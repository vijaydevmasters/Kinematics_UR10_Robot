from sympy.physics.mechanics import dynamicsymbols
import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import scipy as scpy


from sympy.physics.vector import init_vprinting
Axes3D = Axes3D

init_vprinting(use_latex='mathjax', pretty_print=False)


a, alpha, d, theta, theta1, theta2, theta3, theta4, theta5, theta6, l1, a2, a3, l4, l5, l6 = dynamicsymbols(
    'a alpha d theta theta1 theta2 theta3 theta4 theta5 theta6 l1 a2 a3, l4, l5 l6')

# Helper functions


def scos(x): return sp.cos(x).evalf()
def ssin(x): return sp.sin(x).evalf()

# Cross product function


def cross(A, B):
    return [A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0]]

# DH Transformation


def dh_trans(q):

    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]
    theta5 = q[4]
    theta6 = q[5]
    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    t06 = t01 * t12 * t23 * t34 * t45 * t56
    return t06

# DH for Jacobian


def dh_for_jacobian(q):
    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]
    theta5 = q[4]
    theta6 = q[5]

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    T = [t01, t01*t12, t01*t12*t23, t01*t12*t23*t34,
         t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56]
    return T

# Jacobian calculation


def jacobian(T):

    # Z vectors
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append((i[:3, 2]))

    # Origins
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append((i[:3, 3]))

    # Build the Jacobian matrix
    J = sp.zeros(6, 6)

    # The first three rows of the Jacobian are the cross product of z vectors and difference of end-effector and joint origins
    for i in range(6):
        J[0, i] = sp.Matrix(
            cross(z[i], [o[-1][0] - o[i][0], o[-1][1] - o[i][1], o[-1][2] - o[i][2]]))

    # The last three rows of the Jacobian are simply the z vectors for rotational joints
    for i in range(6):
        J[3:6, i] = z[i]
        # sp.pprint(J)
    return J


def print_J_O_Z_vectors():

    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    T = [t01, t01*t12, t01*t12*t23, t01*t12*t23*t34,
         t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56]

  # Z vectors
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append((i[:3, 2]))

    print("****************************************************************************************************")
    print("THE Z-AXES UNIT VECTORS OF LOCAL FRAMES WITH RESPECT TO BASE FRAME: ")
    print("z0:")
    sp.pprint(z[0])

    print("z1:")
    sp.pprint(z[1])

    print("z2:")
    sp.pprint(z[2])

    print("z3:")
    sp.pprint(z[3])

    print("z4:")
    sp.pprint(z[4])

    print("z5:")
    sp.pprint(z[5])

    print("z6:")
    sp.pprint(z[6])

    # Origins
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append((i[:3, 3]))
    print("****************************************************************************************************")
    print("THE ORIGIN VECTORS ARE: ")
    print("O0:")
    sp.pprint(o[0])

    print("O1:")
    sp.pprint(o[1])

    print("O2:")
    sp.pprint(o[2])

    print("O3:")
    sp.pprint(o[3])

    print("O4:")
    sp.pprint(o[4])

    print("O5:")
    sp.pprint(o[5])

    print("O6:")
    sp.pprint(o[6])

    # Build the Jacobian matrix
    J = sp.zeros(6, 6)

    # The first three rows of the Jacobian are the cross product of z vectors and difference of end-effector and joint origins
    for i in range(6):
        J[0, i] = sp.Matrix(
            cross(z[i], [o[-1][0] - o[i][0], o[-1][1] - o[i][1], o[-1][2] - o[i][2]]))

    # The last three rows of the Jacobian are simply the z vect5ors for rotational joints
    for i in range(6):
        J[3:6, i] = z[i]
        # sp.pprint(J)
    print("GENERIC JACOBIAN MATRIX: ")
    sp.pprint(J)


def print_DH_MATRIX():
    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    t06 = t01 * t12 * t23 * t34 * t45 * t56

    print("***********************************************************************************************************")
    print(" Homogenous Transformation Matrix T_06: ")
    sp.pprint(t06)


def draw_circle(q_o, view):

    dt = 0.01
    total_time = np.arange(0, 20, dt)
    omega = 2*np.pi/20
    q = q_o

    x_points = list()
    y_points = list()
    z_points = list()

    for i in total_time:

        x_dot = -sp.N(10*sp.pi*sp.sin(sp.N((sp.pi*i)/10)+sp.pi/2))
        y_dot = 0.0
        z_dot = sp.N(10*sp.pi*sp.cos(sp.N((sp.pi*i)/10)+sp.pi/2))

        epsilon = sp.Matrix([x_dot, y_dot, z_dot, 0, 0, 0])

        T = dh_for_jacobian(q)
        J = jacobian(T)
        j_inv = np.linalg.pinv(np.array(J, dtype=float))
        q_dot = j_inv * epsilon
        q = q + q_dot * dt

        T_end_effector = dh_trans(q)

        x_points.append(T_end_effector[0, 3])
        y_points.append(356.1)
        z_points.append(T_end_effector[2, 3])

    if view == 'isometric view':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(x_points, y_points, z_points)
        ax.set_title("End-Effector Traced Path (ISOMETRIC VIEW)")
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.show()

    elif view == 'front view':
        plt.plot(x_points, z_points)
        plt.title("End-Effector Traced Path (FRONT VIEW)")
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.axis('equal')
        plt.show()

    else:
        print("Enter a valid dimensional view")


# Print Homogenous Transformation Matrix
print_DH_MATRIX()

# Print Origin Vectors, Origin Vectors and Jacobian Matrix
print_J_O_Z_vectors()

# Home Position Of End Effector
q_initial = sp.Matrix([0, 0, 0, 0, 0, 0])

# End Effector Draws Circle
trajectory_points = draw_circle(q_initial, 'isometric view')
