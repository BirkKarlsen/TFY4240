import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

res = 2000
nc_max = 225
res_ar = 0.05

# Fourier Analysis

# This function takes in a numpy-array with values from a function evaluated from 0 to 1, the amount of data points from
# 0 to 1 and the number of coefficients of the Fourier series one wants to have. The function then returns a numpy array
# with the values of the coefficients in the Fourier series of the function that was inputed.
def findCoefficients(Function, res, nc):
    cof = np.zeros(nc)
    x = np.linspace(0, 1, res)
    dx = 1 / res
    for i in range(1, nc + 1):
        sini = np.sin(np.pi * i * x)
        cof[i - 1] = np.sum(Function * sini) * dx * 2 / np.sinh(i * np.pi)
    return cof

# This function takes in a numpy-array of coefficients for a Fourier series, the amount of datapoint desired along the
# x and y axis of the outputed numpy-array and a optional parameter "whole" which evaluates the entire box if True and
# just along the boundary y = L if False. The function returns either a 2D numpy-array with the magnitude of the 2D
# function for each value of x and y in the box or a 1D array with a magnitude of the 2D function along the boundary
# y = L.
def sumSeries(coef, res, whole=True):
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    if whole:
        func = np.zeros((res, res))
        for i in range(1, np.shape(coef)[0] + 1):
            sinhi = np.sinh(i * np.pi * y)
            sini = np.sin(i * np.pi * x)
            func += coef[i - 1] * np.outer(sinhi, sini)
    else:
        func = np.zeros(res)
        for i in range(1, np.shape(coef)[0] + 1):
            sinhi = np.sinh(i * np.pi)
            sini = np.sin(i * np.pi * x)
            func += coef[i - 1] * sinhi * sini
    return func


# Finding the E-field

# This function takes in a 2D numpy-array which represents a 2D potential function. The function returns the negative
# gradient of the 2D function that was inputed.
def findElectricField(V):
    Ey, Ex = np.gradient(V)
    return -Ex, -Ey


# Plotting Functions

# This function takes in two lists, which are each 2D numpy-arrays that represent the magnitude in two direction of the
# electric field. The last parameter is the density of vectors of the two 2D arrays that are returned. The function
# returnes two 2D arrays that only have a selected amount of points from the two arrays that was taken in.
def EForPlotting(Ex, Ey, res_ar):
    res = np.shape(Ex)[0]
    Exp = Ex[::int(res * res_ar), ::int(res * res_ar)]
    Eyp = Ey[::int(res * res_ar), ::int(res * res_ar)]
    return Exp, Eyp

# This function takes in a 2D array that represent the electric potential, the name of the function for the y = L
# boundary and the resolution of the colorbar in the contourplot. The function then plots the potential as a contour
# plot.
def plotContourfilled(V, funcname, cres=99):
    res = np.shape(V)[0]
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca()
    clev = np.linspace(V.min(), V.max(), cres)
    cr = ax.contourf(X, Y, V, clev, cmap=plt.cm.coolwarm)
    ax.set_xlabel('$x / L$ [-]')
    ax.set_ylabel('$y / L$ [-]')
    ax.set_title('$V(x,y)$ for ' + funcname)
    cbar = fig.colorbar(cr)
    cbar.set_label('$V(x,y)/V_{c}$ [-]')
    plt.show()

# This function takes in two 2D arrays that represent the magnitude of the electric field in the x- and y-directions,
# respevtively. The function also takes in a string which is the name of the function at the y = L boundary, the
# amount of vectors along each axis of the plot, cmax is a max value of the magnitude of field, realLength is a bool
# variable that determines whether or not to plot the actual length of the vectors, cont is a bool varibale that
# determines whether or not to have a contourplot behind the vectorplot to indicate strength of the field, cres
# is the resolution of the colorbar of the contourplot and g is a varibale that determines the scaling of the colors
# to the values of the magnitude.
def plotVectorfield(Ex, Ey, funcname, res_ar, cmax=None, realLength=False, cont=True, cres=99, g=1):
    res = np.shape(Ex)[0]
    X, Y = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
    fig = plt.figure()
    ax = fig.gca()

    ax.set_title('$E(x,y)$ for ' + funcname)

    ax.set_xlabel('$x / L$ [-]')
    ax.set_ylabel('$y / L$ [-]')

    if cont:
        Emag = np.sqrt(Ex ** 2 + Ey ** 2)
        if cmax == None:
            clev = np.linspace(Emag.min(), Emag.max(), cres)
        else:
            clev = np.linspace(Emag.min(), cmax, cres)
        cr = ax.contourf(X, Y, Emag, clev, cmap=plt.cm.coolwarm, norm=colors.PowerNorm(gamma=g))
        cbar = fig.colorbar(cr)
        cbar.set_label('$E(x,y)/E_c$ [-]')

    X, Y = np.meshgrid(np.arange(res_ar, 1, res_ar), np.arange(res_ar, 1, res_ar))
    Exp, Eyp = EForPlotting(Ex, Ey, res_ar)

    if realLength:
        ax.quiver(X, Y, Exp[1:,1:], Eyp[1:,1:])
    else:
        Exnorm = Exp / np.sqrt(Exp ** 2 + Eyp ** 2)
        Eynorm = Eyp / np.sqrt(Exp ** 2 + Eyp ** 2)
        ax.quiver(X, Y, Exnorm[1:,1:], Eynorm[1:,1:])

    plt.show()

# This function plots the boundaries of the potential with the boundary conditions in four subplots, one for each
# boundary. it takes in the function for y = L, the potential, the name of the function at y = L and a bool value that
# determines whether or not to plot all the boundaries or just at y = L.
def plotBoundary(Func, ApproxFunc, funcname, all=False):
    x = np.linspace(0, 1, np.shape(Func)[0])
    ze = np.zeros(np.shape(Func)[0])
    if all:
        plt.suptitle('Boundaries for ' + funcname)

        plt.subplot(221)
        plt.title('For $y=L$')
    else:
        plt.title('$y=L$ for ' + funcname)

    plt.plot(x, Func, label='$V_{0}$')
    plt.plot(x, ApproxFunc[-1,:], label='Approximation')
    plt.xlabel('$x/L$ [-]')
    plt.ylabel('$V(x,L)/V_c$ [-]')

    if all:
        plt.subplot(222)
        plt.title('For $y=0$')
        plt.plot(x, ze, label='$V = 0$')
        plt.plot(x, ApproxFunc[0,:], label='Approximation')
        plt.xlabel('$x/L$ [-]')
        plt.ylabel('$V(x,0)/V_c$ [-]')

        plt.subplot(223)
        plt.title('For $x=L$')
        plt.plot(x, ze, label='$V = 0$')
        plt.plot(x, ApproxFunc[:,-1], label='Approximation')
        plt.xlabel('$y/L$ [-]')
        plt.ylabel('$V(L,y)/V_c$ [-]')

        plt.subplot(224)
        plt.title('For $x=0$')
        plt.plot(x, ze, label='$V = 0$')
        plt.plot(x, ApproxFunc[:,0], label='Approximation')
        plt.xlabel('$y/L$ [-]')
        plt.ylabel('$V(0,y)/V_c$ [-]')

    plt.show()


# Functions for V(x)
def SineFunc(m, x):
    return np.sin(m * np.pi * x)

def PolyFunc(x):
    return 1 - (x - 0.5)**4

def HeavieSide(x):
    length = np.shape(x)[0]
    func = np.zeros(length)
    for i in range(length):
        if i > length // 2 and i < 3 * length // 4:
            func[i] = 1
    return func


# Check for convergence

# This function takes in the function at y = L and the potential at y = L and summes up the absolute difference between
# each point along the boundary y = L.
def integrateDifference(Func, AppFunc):
    dx = 1 / np.shape(Func)[0]
    return np.sum(np.abs(Func - AppFunc)) * dx

# This function find the the difference between numpyarray of a function at y = L (Func) and the Fourier series of this
# function. The function returns an array with where element i is the difference between the actual function and the
# Fourier series up to term i + 1.
def differenceAsFuncNCoeff(Func, Coefs):
    res = np.shape(Func)[0]
    diff = np.zeros(np.shape(Coefs))
    for i in range(1,np.shape(Coefs)[0] + 1):
        cofi = Coefs[:i]
        AppFunc = sumSeries(cofi, res, whole=False)
        diff[i - 1] = integrateDifference(Func, AppFunc)
    return diff

# This function plots the absolute difference between the different functions and its Fourier series for the boundary
# y = L as a function of the terms included in the Fourier series. The functions takes in the parameters res and nc,
# which are the amount of datapoints along the boundary and the maximum amount of terms in the Fourier series,
# respectively.
def CompareConvergence(res, nc):
    x = np.linspace(0, 1, res)
    n = np.linspace(1, nc, nc)

    sin1 = SineFunc(1, x)
    sin100 =SineFunc(100, x)
    polyf = PolyFunc(x)
    heavief = HeavieSide(x)

    data = np.array([sin1, sin100, polyf, heavief])

    datadiff = np.zeros((4,nc))

    for i in range(np.shape(data)[0]):
        cofi = findCoefficients(data[i,:], res, nc)
        datadiff[i,:] = differenceAsFuncNCoeff(data[i,:], cofi)

    plt.title('Difference as a function of number of terms')
    plt.plot(n, datadiff[0,:], label='Sinues, $m=1$')
    plt.plot(n, datadiff[1,:], label='Sinus, $m=100$')
    plt.plot(n, datadiff[2,:], label='Polynomial')
    plt.plot(n, datadiff[3,:], label='Heavieside')

    plt.xlabel('Number of terms')
    plt.ylabel('Difference from actual function')
    plt.xlim(1, nc)
    plt.legend()
    plt.show()
