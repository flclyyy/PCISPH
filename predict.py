import main

def velocity(predict_list, pressure_force, other_force, i, dt, m0):
    ax = (pressure_force[i][1] + other_force[i][1]) / m0
    ay = (pressure_force[i][2] + other_force[i][2]) / m0
    az = (pressure_force[i][3] + other_force[i][3]) / m0
    vx = predict_list[i][4] + ax * dt
    vy = predict_list[i][5] + ay * dt
    vz = predict_list[i][6] + az * dt
    return vx, vy, vz


def position(predict_list, i, dt):
    x = predict_list[i][1] + predict_list[i][4] * dt
    y = predict_list[i][2] + predict_list[i][5] * dt
    z = predict_list[i][3] + predict_list[i][6] * dt
    return x, y, z


def density(predict_list, i):
    return 0


def variation(predict_density, i):
    density0 = 0
    return predict_density[i] - density0