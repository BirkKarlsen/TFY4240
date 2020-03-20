
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.gca(projection='3d')
r_earth = 6371 * 10**3

x, y, z = np.meshgrid(np.arange(-2 *r_earth, 2*r_earth, 0.5 * r_earth), np.arange(-2*r_earth, 2.1*r_earth, 0.5 * r_earth), np.arange(-2*r_earth, 2.1*r_earth, 0.5 * r_earth))


# Find the values in polarcoordinates
theta = np.arctan(np.sqrt(x**2 + y**2) / z)
phi = np.arctan(y / x)
r = np.sqrt(x**2 + y**2 + z**2)

# constants
my0 = 1.2566 * 10**(-6)
m = 8e22

k0 = (m * my0) / (4 * np.pi)

u = (3 * k0 / 2) * np.cos(phi) * np.sin(2 * theta)
v = (3 * k0 / 2) * np.sin(phi) * np.sin(2 * theta)
w = k0 * (3 * np.cos(theta)**2 - 1)

ax.quiver(x, y, z, u, v, w, length=0.1, color='black')

plt.show()

x, z = np.meshgrid(np.arange(-2, 2.1, 0.5), np.arange(-2, 2.1, 0.5))

theta = np.arctan(x / z)
r = np.sqrt(x**2 + z**2)

u = (3 * k0 / 2) * np.sin(2 * theta)
w = k0 * (3 * np.cos(theta) **2 - 1)

circle_res = np.arange(0, 2 * np.pi, 0.01)
radi = 0.6

fig, ax = plt.subplots(figsize=(6, 6))

ax.streamplot(x, z, u, w)
ax.set_aspect('equal')
#ax.plot(0,0,'-or')
#for i in circle_res:
#    ax.plot(radi * np.cos(i), radi * np.sin(i), '-ob')
ax.set_title('Stream Plot of Two Point Charges')

plt.show()