import create_mode
import neighbor
import force
import predict
import numpy as np

# init
particle_list = create_mode.create()

t0 = 0  # the max time
t = 0
eit = 0
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

def delta(predict_list):
    return 0

while t < t0:
    for i in range(length):
        neighbors_index[i] = neighbor.find_neighbors(i, particle_list)
        other_force[i] = force.totel(i, particle_list)
    k = 0
    while max(density_variation) > eit or k < 3 :
        predict_list = particle_list
        for i in range(length):
            predict_vx[i], predict_vy[i], predict_vz[i] = predict.velocity(predict_list, pressure_force, other_force, i)
            predict_x[i], predict_y[i], predict_z[i] = predict.position(predict_list, i)
            predict_list[i] = zip(predict_x[i], predict_y[i], predict_z[i], predict_vx[i], predict_vy[i], predict_vz[i])
        for i in range(length):
            predict_density[i] = predict.density(predict_list, i)
            density_variation[i] = predict.variation(predict_density, i)
            p[i] = p[i] + delta(predict_list) * density_variation[i]
        for i in range(length):
            pressure_force[i] = force.pressure(predict_list, p, i)
        k = k + 1
    particle_list = predict_list





