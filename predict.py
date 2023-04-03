import main
import math

def distant(predict_list, index, i, j):
    x_ij = predict_list[i][0] - predict_list[index[j]][0]
    y_ij = predict_list[i][1] - predict_list[index[j]][1]
    z_ij = predict_list[i][2] - predict_list[index[j]][2]
    r = math.sqrt(x_ij ** 2 + y_ij ** 2 + z_ij ** 2)
    return r


def velocity(predict_list, pressure_force, other_force, i, dt, m0):
    ax = (pressure_force[i][0] + other_force[i][0]) / m0
    ay = (pressure_force[i][1] + other_force[i][1]) / m0
    az = (pressure_force[i][2] + other_force[i][2]) / m0
    vx = predict_list[i][3] + ax * dt
    vy = predict_list[i][4] + ay * dt
    vz = predict_list[i][5] + az * dt
    return vx, vy, vz


def position(predict_list, i, dt):
    x = predict_list[i][0] + predict_list[i][3] * dt
    y = predict_list[i][1] + predict_list[i][4] * dt
    z = predict_list[i][2] + predict_list[i][5] * dt
    return x, y, z


def density(predict_list, index, i, m0, h):
    pre_density = 0
    for j in range(len(index)):
        r = distant(predict_list, index, i, j)
        pre_density = pre_density + m0 * main.W(r, h)
    return pre_density


def variation(predict_density, i):
    density0 = 0
    return predict_density[i] - density0