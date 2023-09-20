import numpy as np

from numpy import cos, sin

L1 = 0.055
L2 = 0.315
L3 = 0.045
L4 = 0.108
L5 = 0.005
L6 = 0.034
L7 = 0.015
L8 = 0.088
L9 = 0.204


def A_1_to_0(th_1: float, X: np.array):
    x = X[0]
    y = X[1]
    z = X[2]
    return np.array(
        [
            (x + L6) * cos(th_1) - (-z - L4) * sin(th_1),
            (x + L6) * sin(th_1) + (-z - L4) * cos(th_1),
            y + L2 + L3,
            1,
        ]
    )

def A_2_to_1(th_2: float, X: np.array):
    x = X[0]
    y = X[1]
    z = X[2]
    return np.array(
        [
            (x + L7) * cos(th_2) - (y - L8) * sin(th_2),
            (x + L7) * sin(th_2) + (y - L8) * cos(th_2),
            z - L5,
            1,
        ]
    )

def A_3_to_2(th_3: float, X: np.array):
    x = X[0]
    y = X[1]
    z = X[2]
    return np.array(
        [
            x * cos(th_3) - (y - L9) * sin(th_3),
            x * sin(th_3) + (y - L9) * cos(th_3),
            z,
            1
        ]
    )

def A_3_to_0(th_1: float, th_2: float, th_3: float, X:np.array):
    return A_1_to_0(th_1, A_2_to_1(th_2, A_3_to_2(th_3, X)))
