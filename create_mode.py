import numpy as np

def create():
    num_points = 100
    spacing = 1.0
    x, y, z = np.meshgrid(np.arange(0, num_points*spacing, spacing),
                          np.arange(0, num_points*spacing, spacing),
                          np.arange(0, num_points*spacing, spacing),)

    totel = num_points ** 3
    vx, vy, vz = np.zeros(totel), np.zeros(totel), np.zeros(totel)

    points_list = list(zip(x.ravel(), y.ravel(), z.ravel(), vx, vy, vz))
    return points_list