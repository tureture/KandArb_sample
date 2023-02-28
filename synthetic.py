# Plottar den syntetiska datan

import numpy as np
from os.path import join
import nibabel as nib
import GPy
import torch
import gpytorch
import dipy
from scipy.spatial.transform import Rotation as R
import heapq
import itertools
import cProfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def E(q,phi=0):
    # Returnerar syntetiska värdet för en viss punkt i q-space
    q = np.array(q)
    D0 = 2.5e-9
    D0 = 10
    D1 = np.diag([1, 0.1, 0.1]) * D0
    theta = np.radians(phi)
    D2 = np.diag([1*np.cos(theta)+0.1*np.sin(theta), 1*np.sin(theta)+0.1*np.cos(theta), 0.1]) * D0
    qt = np.transpose(q)
    td = 0.01
    return 0.5*(np.exp(-td*qt.dot(D1).dot(q))+np.exp(-td*qt.dot(D2).dot(q)))


grid = np.zeros((101,101,101))
for rad in range(101):
    for kolon in range(101):
        for djup in range(101):
            x = 1*rad - 50
            y = 1*kolon - 50
            z = 1*djup - 50
            grid[rad,kolon,djup] = E((x,y,z),90)
gri = grid[:,:,0]
four = np.fft.ifftshift(np.abs(np.real(np.fft.ifftn(grid))))

x = np.linspace(-50, 50, 101)
y = np.linspace(-50, 50, 101)
X, Y = np.meshgrid(x, y)
Z = four[:,:,0]
print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()