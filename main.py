import create_mode
import neighbor
import force
import predict
import numpy as np

# init
particle_list = create_mode.create()

m0 = 0  # the mass of one particle
t0 = 0  # the max time
t = 0
eit = 0
h = 0
density0 = 0
length = len(particle_list)

p = np.zeros(length)
density = np.zeros(length)
predict_density = density
density_variation = density

predict_vx, predict_vy, predict_vz = np.zeros(length), np.zeros(length), np.zeros(length)
predict_x, predict_y, predict_z = np.zeros(length), np.zeros(length), np.zeros(length)
predict_list = list(zip(predict_x, predict_y, predict_z, predict_vx, predict_vy, predict_vz))


Fx, Fy, Fz = np.zeros(length), np.zeros(length), np.zeros(length)
pressure_force = list(zip(Fx, Fy, Fz))
other_force = pressure_force

neighbors_index = np.empty(length, dtype = object)


# 三次B样条核函数其他文件也要用
def W(r, h):
        return 0


def gradient_W(r, h):
    return 0, 0, 0


def delta(particle_list, index, dt, density0, m0, i):
    beta = 2 * (m0 * dt / density0) ** 2
    sum0 = 0
    sum_x, sum_y, sum_z = 0, 0, 0
    for j in range(len(index)):
        r = predict.distant(particle_list, index, i, j)
        result = gradient_W(r, h)
        sum0 = sum0 + np.dot(result, result)
        sum_x = sum_x + result[0]
        sum_y = sum_y + result[0]
        sum_z = sum_z + result[0]
    ret = -1/(beta*(-(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)-sum0))
    return ret


while t < t0:
    dt = 0
    for i in range(length):
        neighbors_index[i] = neighbor.find_neighbors(i, particle_list)
    for i in range(length):
        other_force[i] = force.totel(i, particle_list, neighbors_index[i], p, m0)
    k = 0
    while max(density_variation) > eit or k < 3:
        predict_list = particle_list
        for i in range(length):
            predict_vx[i], predict_vy[i], predict_vz[i] = predict.velocity(predict_list, pressure_force, other_force, i, dt, m0)
            predict_x[i], predict_y[i], predict_z[i] = predict.position(predict_list, i, dt)
            predict_list[i] = zip(predict_x[i], predict_y[i], predict_z[i], predict_vx[i], predict_vy[i], predict_vz[i])
        for i in range(length):
            # we have updated the predicted position , so we don't need to update the distances to neighbors
            predict_density[i] = predict.density(predict_list, neighbors_index[i], i, m0, h)
            density_variation[i] = predict.variation(predict_density, i)
            p[i] = p[i] + delta(predict_list, neighbors_index[i], dt, density0, m0, i) * density_variation[i]
        for i in range(length):
            pressure_force[i] = force.pressure(predict_list, neighbors_index[i], p, i, m0, h)
        k = k + 1
    particle_list = predict_list
    t = t + dt





