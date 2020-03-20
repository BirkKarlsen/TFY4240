import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

L = 20
n = 20000
k = 1
freq = k / L

def findCoefficients(Function, L, res, nc):
    cof = np.zeros((nc, 2))
    x = np.linspace(0, L, res)
    dx = L / res
    for i in range(nc):
        sini = np.sin(2 * np.pi * i * x / L)
        cosi = np.cos(2 * np.pi * i * x / L)
        cof[i,0] = np.sum(Function * cosi) * dx * (2 / L)
        cof[i,1] = np.sum(Function * sini) * dx * (2 / L)
    cof[0,:] = cof[0,:] / 2
    return cof

def plotCoefficients(cof, L, res):
    nc = np.shape(cof)[0]
    x = np.linspace(0, L, res)
    func = np.zeros(res)
    for i in range(nc):
        sini = np.sin(2 * np.pi * i * x / L)
        cosi = np.cos(2 * np.pi * i * x / L)
        func += cof[i,0] * cosi + cof[i,1] * sini
    return func

def squareWave(x, A):
    length = np.shape(x)[0]
    func = np.zeros(length)
    for i in range(length // 2, length):
        func[i] = A
    return func

x = np.linspace(0, L, n)
sqrW = squareWave(x, 1)

coefs = findCoefficients(sqrW, L, n, 10000)
Approx = plotCoefficients(coefs, L, n)

plt.plot(x, sqrW)
plt.plot(x, Approx)
plt.show()
