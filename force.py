import main
import predict


def pressure(predict_list, index, p, i, m0, h):
    Fx, Fy, Fz = 0, 0, 0
    for j in range(len(index)):
        xij = predict.distant(predict_list, index, i, j)
        temp = main.gradient_W(xij, h)
        result = [- m0 * m0 * (p[i]/predict.density(predict_list, index, i, m0, h) + p[j]/predict.density(predict_list, index, j, m0, h)) * k for k in temp]
        Fx = Fx + result[0]
        Fy = Fy + result[1]
        Fz = Fz + result[2]
    return Fx, Fy, Fz


def surface_tension():  # 可以先放着
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


def viscous_force():  # 可以先放着
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


def volumetric_force(predict_list, i, m0):
    g = 9.8
    Fx, Fy, Fz = 0, 0, m0*g
    return Fx, Fy, Fz


# this function add all forces expect pressure
def totel(i, list, predict_list, p, m0):
    result1 = surface_tension()
    result2 = viscous_force()
    result3 = volumetric_force(list, i, m0)
    Fx = result1[0] + result2[0] + result3[0]
    Fy = result1[1] + result2[1] + result3[1]
    Fz = result1[2] + result2[2] + result3[2]
    return Fx, Fy, Fz
