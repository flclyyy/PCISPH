import sys

import taichi as ti
import numpy as np
import math

# global constant
dimension = 2
numPreLine = 40
numPar = numPreLine ** dimension
d = 0.2
kernelRadius = 2 * d
density0 = 1 / d ** dimension
mass = 1.0
boundX = 15.0
boundY = 15.0
boundZ = 15.0
restiCoeff = 0.8
rest = 0.9
# fricCoeff = 0.1
EosCoeff = 200.0
EosExponent = 7.0
mu = 4
dt = 1e-3
numStatic = int(6 * (boundY/d + boundX/d - 4))
numTotal = numStatic + numPar
cellSize = 2 * kernelRadius
numCellX = ti.ceil(boundX/cellSize)
numCellY = ti.ceil(boundY/cellSize)
#numCellZ = ti.ceil(boundZ/cellSize)
numCell = numCellX * numCellY #* numCellZ

ti.init(arch=ti.gpu)

# physical fields
position = ti.Vector.field(dimension, float, shape=numTotal)
velocity = ti.Vector.field(dimension, float, shape=numPar)
density = ti.field(float, shape=numTotal)
pressure = ti.field(float, shape=numPar)
acceleration = ti.Vector.field(dimension, float, shape=numPar)
pressureGradientForce = ti.Vector.field(dimension, float, shape=numPar)
viscosityForce = ti.Vector.field(dimension, float, shape=numPar)


# neighbor search variables
maxNumNeighbors = 1000
maxNumParInCell = 1000

numParInCell = ti.field(int, shape=numCell)
neighbor = ti.field(int, shape=(numTotal, maxNumNeighbors))
numNeighbor = ti.field(int, shape=numTotal)
cell2Par = ti.field(int, shape=(numCell, maxNumParInCell))
numCell2Par = ti.field(int, shape=numTotal)

# kernel Func
@ti.func
def kernelFunc(r_norm):
    res = ti.cast(0.0, ti.f32)
    h = kernelRadius
    k = 1.0
    if dimension == 3:
        k = 8 / np.pi
    elif dimension == 2:
        k = 40/7/np.pi
    k /= h ** 3
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res


@ti.func
def cubic_kernel_derivative(r):
    h = kernelRadius
    k = 1.0
    if dimension == 3:
        k = 8 / np.pi
    elif dimension == 2:
        k = 40/7/np.pi
    k = 6. * k / h ** 3
    r_norm = r.norm()
    q = r_norm / h
    #print("r_norm q",r_norm,q)
    res = ti.Vector([0.0 for _ in range(dimension)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    #print("res",res)
    return res

# neighbor search
@ti.func
def getCell(pos):
    cellID = int(pos.x/cellSize) + \
        int(pos.y/cellSize) * numCellX
        #int(pos.z/cellSize) * numCellX * numCellY
    return cellID


@ti.func
def IsInBound(c):
    return 0 <= c < numCell


@ti.kernel
def neighborSearch():
    for par in range(numPar):
        cell = getCell(position[par])
        k = ti.atomic_add(numParInCell[cell], 1)
        cell2Par[cell, k] = par
        #print((k))

    for i in range(numPar):
        cell = getCell(position[i])
        kk = 0
        '''
        offs = ti.Vector([0] * 27)
        for k in range(3):
            for m in range(3):
                for n in range(3):
                    index = k * 9 + \
                            m * 3 + n
                    offs[index] = (k - 1) * numCellX * numCellY + \
                            (m - 1) * numCellX + (n - 1)
        '''
        offs = ti.Vector([0] * 9)
        for m in range(3):
            for n in range(3):
                index = m * 3 + n
                offs[index] = (m - 1) * numCellX + (n - 1)
        neiCellList = cell + offs

        for ii in ti.static(range(9)):
            cellToCheck = neiCellList[ii]
            #print(cellToCheck)
            if IsInBound(cellToCheck):
                for k in range(numParInCell[cellToCheck]):
                    j = cell2Par[cellToCheck, k]
                    temp = (position[i] - position[j]).norm()
                    #print(temp, kernelRadius)
                    if kk < maxNumNeighbors and j != i and \
                             temp < 1.001*kernelRadius:
                        neighbor[i, kk] = j
                        kk += 1

        numNeighbor[i] = kk

# physical compute
@ti.kernel
def computeDensity():
    for i in range(numPar):
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]
            r = (position[i] - position[j]).norm()
            density[i] += mass * kernelFunc(r)

        if density[i] < density0:
            density[i] = density0

    for i in range(numStatic):
        density[i+numPar] = density0

@ti.kernel
def computeViscosityForce():
    for i in viscosityForce:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]
            r = position[i] - position[j]
            v_xy = (velocity[i] - velocity[j]).dot(r)
            viscosityForce[i] += 2 * (2 + dimension) * mu * d**dimension * (mass / density[j]) * v_xy / (r.norm()**2 + 0.01 * kernelRadius**2) * cubic_kernel_derivative(r)


@ti.kernel
def computePressure():
    for i in pressure:
        pressure[i] = EosCoeff*((density[i]/density0) ** EosExponent - 1.0)


@ti.kernel
def computePressGradientForce():
    for i in pressureGradientForce:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]
            r = position[i] - position[j]
            pressureGradientForce[i] -= mass * mass * (
                    pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2) * cubic_kernel_derivative(r)


@ti.kernel
def computeAcceleration():
    for i in acceleration:
        gravity = ti.Vector([0, -100])
        acceleration[i] += gravity + viscosityForce[i]/mass + pressureGradientForce[i]/mass


@ti.kernel
def advanceTime():
    for i in range(numPar):
        velocity[i] += acceleration[i] * dt
        position[i] += velocity[i] * dt


@ti.kernel
def boundaryCollision():
    for i in range(numPar):
        if position[i].x < 0.0:
            position[i].x = 0.0
            velocity[i].x *= -restiCoeff
            velocity[i].y *= rest

        elif position[i].x >= boundX:
            position[i].x = boundX
            velocity[i].x *= -restiCoeff
            velocity[i].y *= rest

        elif position[i].y < 0.0:
            position[i].y = 0.0
            velocity[i].y *= -restiCoeff
            velocity[i].x *= rest

        elif position[i].y >= boundY:
            position[i].y = boundY
            velocity[i].y *= -restiCoeff
            velocity[i].x *= rest
'''
        if dimension == 3:
            if position[i].z < 0.0:
                position[i].z = 0.0
                velocity[i].z *= -restiCoeff

            if position[i].z >= boundZ:
                position[i].z = boundZ
                velocity[i].z *= -restiCoeff
'''

@ti.func
def simualteCollisions(i, vec, l):
    c_f = 0.5
    position[i] += vec * l
    velocity[i] -= (1.0 + c_f) * velocity[i].dot(vec) * vec


@ti.kernel
def enforceBoundary():
    for p_i in range(numPar):
        pos = position[p_i]
        if pos[0] < 3*d:
            simualteCollisions(p_i, ti.Vector([1.0, 0.0]), 3*d - pos[0])
        if pos[0] > boundX - 3*d:
            simualteCollisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - (boundX - 3*d))
        if pos[1] > boundY - 3*d:
            simualteCollisions(p_i, ti.Vector([0.0, -1.0]), pos[1] - (boundY - 3*d))
        if pos[1] < 3*d:
            simualteCollisions(p_i, ti.Vector([0.0, 1.0]), 3*d - pos[1])


def clear():
    # clear the density
    density.fill(0.0)

    # clear the forces and acceleration
    acceleration.fill(0.0)
    pressureGradientForce.fill(0.0)
    viscosityForce.fill(0.0)

    # clear the neighbor list and cell2Par
    numParInCell.fill(0)
    numNeighbor.fill(0)
    neighbor.fill(-1)
    cell2Par.fill(0)


def step():

    clear()
    neighborSearch()
    computeDensity()
    #print("Density", density[500])
    computeViscosityForce()
    #print("Vis", viscosityForce[500][0],viscosityForce[500][1],viscosityForce[500][2])
    computePressure()
    #print("Pre", pressure[500])
    computePressGradientForce()
    #print("PreForce", pressureGradientForce[500][0], pressureGradientForce[500][1], pressureGradientForce[500][2])
    computeAcceleration()
    #print("Ac",acceleration[500][0],acceleration[500][1],acceleration[500][2])
    advanceTime()
    #boundaryCollision()
    enforceBoundary()



def initialization():
    for i in range(numPar):
        position[i] = [
            i % numPreLine * d + 4*d,
            (i // numPreLine) * d + 4*d,
            #i // (numPreLine ** 2) * d
        ]
        #print(position[i][0], position[i][1])

    k = 0
    # num = 3 * (boundX / d + 1)
    for i in range(3):
        y = boundY + (i-2) * d
        x = 0.0
        while x <= boundX:
            index = numPar + k
            position[index] = [x, y]
            x += d
            k += 1

    # 3*(boundY/d - 5)
    for i in range(3):
        x = d * i
        y = boundY - 3 * d
        while y >= 3 * d:
            index = numPar + k
            position[index] = [x, y]
            y -= d
            k += 1

    # 3*(boundY/d - 5)
    for i in range(3):
        x = boundX - (2-i) * d
        y = boundY - 3 * d
        while y >= 3 * d:
            index = numPar + k
            position[index] = [x, y]
            y -= d
            k += 1

    # num = 3 * (boundX/d + 1)
    for i in range(3):
        y = (2-i) * d
        x = 0.0
        while x <= boundX:
            index = numPar + k
            position[index] = [x, y]
            x += d
            k += 1



def draw(gui):
    pos = position.to_numpy()
    #pos[:, 0] *= 1.0 / boundX
    #pos[:, 1] *= 1.0 / boundY
    pos1 = np.zeros((numPar, 2))
    for i in range(numPar):
        pos1[i][0] = position[i][0] * 1.0 / boundX
        pos1[i][1] = position[i][1] * 1.0 / boundY

    pos2 = np.zeros((numStatic, 2))
    for i in range(numStatic):
        pos2[i][0] = position[i+numPar][0] * 1.0 / boundX
        pos2[i][1] = position[i+numPar][1] * 1.0 / boundY
    # draw the particles
    gui.circles(pos1,
                radius=3.0,
                )
    gui.circles(pos2,
                radius=3.0,
                color=0x068587
                )

    # highlight particle 0 with red

    gui.text(
        content=f'press space to pause',
        pos=(0, 0.99),
        color=0x0)

    gui.text(content="position[0]={}".format(position[0]),
             pos=(0, 0.95),
             color=0x0)


def run():
    gui = ti.GUI("SPHDamBreak",
                background_color=0x112F41,
                res=(500, 500)
                 )
    while True:
        step()
        draw(gui)
        gui.show()


initialization()
#print(position[numPar + 100])
run()
