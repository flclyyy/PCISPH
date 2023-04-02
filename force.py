

def pressure(list, p, i):
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


def surface_tension():  # 可以先放着
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


def viscous_force():  # 可以先放着
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


def volumetric_force(list, i):
    Fx, Fy, Fz = 0, 0, 0
    return Fx, Fy, Fz


# this function add all forces expect pressure
def totel(i, list, index, p):
    result1 = pressure(list, p, i)
    result2 = surface_tension()
    result3 = viscous_force()
    result4 = volumetric_force(list, i)
    Fx = result1[1] + result2[1] + result3[1] + result4[1]
    Fy = result1[2] + result2[2] + result3[2] + result4[2]
    Fz = result1[3] + result2[3] + result3[3] + result4[3]
    return Fx, Fy, Fz
