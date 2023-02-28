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
import multiprocessing as mp
import heapq
import time
"""""
Kör första delen på flera kärnor samtidigt. Användbar när en dator men många kärnor används och mängden punkter är många.
"""""
def E(q,phi=90):
    # Returnerar syntetiska värdet för en viss punkt i q-space
    q= np.array(q)
    #D0 = 2.5e-9
    #D0 = 10
    D0 = 0.01
    D1 = np.diag([1, 0.1, 0.1]) * D0
    theta = np.radians(phi)
    D2 = np.diag([1*np.cos(theta)+0.1*np.sin(theta), 1*np.sin(theta)+0.1*np.cos(theta), 0.1]) * D0
    qt = np.transpose(q)
    td = 0.01
    return 0.5*(np.exp(-td*qt.dot(D1).dot(q))+np.exp(-td*qt.dot(D2).dot(q)))

def Mesh(n):
    # Skapar meshen över de q-space punkter vi vill testa
    # Eftersom E(q) = E(-q) så behöver vi bara kolla på halva rummet. Däran n//2 på x.
    # Borde öka prestandan med x8. Kan optimeras yttligare genom att ha varierande stegstorlek.
    space = 1
    #x = np.linspace(0,space,n//2+1) 
    x = np.linspace(-space,space,n)
    y = np.linspace(-space,space,n)
    z = np.linspace(-space,space,n)
    return torch.tensor(list(itertools.product(x, y, z)))


def diff_val(y,S,A,AAinv,Kern):
    y_re = y.reshape(1,-1)
    total_sensor = torch.cat((A,y_re), 0)
    Se = S[~S.unsqueeze(1).eq(total_sensor).all(-1).any(-1)]
    SS = Kern(Se).evaluate() #+ torch.diag(torch.randn(Se.shape[0])/(100000)) # Lägger till brus för att göra den inverterbar
    SSinv = torch.inverse(SS)
    vy = Kern(y_re).evaluate()

    # Skapar yA covariance matrisen
    yA = 't'    # Ful lösning men funkar
    for i in A:
        if yA == 't':
            yA = Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)
        else:
            yA = torch.cat((yA, Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)),0)

    yAt = torch.t(yA)
    yS = []
    for i in Se:
        red = i.reshape(1,-1)
        new = Kern.forward(y_re,red)
        yS.append(new)
    yS = torch.tensor(yS).reshape(-1,1)
    ySt = torch.t(yS)
    diff = (vy-torch.matmul(yAt.float(),torch.matmul(AAinv.float(), yA.float())))/(vy-torch.matmul(ySt.float(),torch.matmul(SSinv.float(),yS.float())))
    return diff

def initi(y,S,A,AAinv,Kern,q):
    q.append([diff_val(y,S,A,AAinv,Kern).detach(), y])

if __name__ == '__main__':
    # Skapar default kernel, ska ändras till kernel som beskrivs i Sjölund vid senare tillfälle.
    Kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    n = 11
    S = Mesh(n)
    A = torch.tensor([[0, 0, 0]]) # Skapar en utgångspunkt i 0,0,0 att arbeta från.
    cand = S[~S.unsqueeze(1).eq(A).all(-1).any(-1)] # Skapar delmängden S\A, alltså platser för nästa sensor
    q = []
    manager = mp.Manager()
    lis = manager.list()
    AA = Kern(A).evaluate() # Skapar vår AA covariance matris
    AAinv = torch.inverse(AA) # Inverterar den.
    any = AAinv.detach()
    inputs = []
    """""
    tic = time.perf_counter()
    print(f'Starting initial evaluation for approximatley {(n**3)/2} points')
    for i in cand:
        inputs.append((i,S,A,any,Kern,lis))
    p = mp.Pool(mp.cpu_count())
    p.starmap(initi,inputs)
    p.close()
    p.join()
    for i in lis:
        heapq.heappush(q, (-i[0], i[1]))
    toc = time.perf_counter()
    print(f'Initial evaluation finished!\nTime: {toc-tic}')
    # Loop som ger oss de bästa sensor placeringarna.
    num_sensor = 100
    print(f'Starting placement of {num_sensor} sensors')
    for k in range(num_sensor):
        AA = Kern(A).evaluate() # Skapar vår AA covariance matris
        AAinv = torch.inverse(AA) # Inverterar den.
        run = True
        current = set()
        while run == True:
            diff_old, y = heapq.heappop(q)
            if y in current:
                run = False
            diff = diff_val(y,S,A,AAinv,Kern)
            if run == False:
                besty = y
            else:
                heapq.heappush(q, (-diff, y))
                current.add(y)
        A = torch.cat((A,besty.reshape(1,-1)), 0)
        print(f'Sensor {k+1} placed')
    toc2 = time.perf_counter()
    print(f'Finished sensor placement!\nTime for section: {toc2-toc}')
    print(A)
    # Add symmetric parts
    #for sensor in A:
    #    inv = torch.tensor((-sensor[0], sensor[1], sensor[2]))
    #    A = torch.cat((A,inv.reshape(1,-1)), 0)
    """""
    scale = 1000
    grid = np.zeros((n,n,n))
    for sensor in S:
        x = int(round(float((sensor[0]+1)*(n-1)/2)))
        y = int(round(float((sensor[1]+1)*(n-1)/2)))
        z = int(round(float((sensor[2]+1)*(n-1)/2)))
        val = E(sensor*scale,90)
        if grid[x,y,z] != 0:
            print('Duplicate detected')
            print(sensor)
        grid[x,y,z]= val

    four = np.fft.ifftshift(np.abs(np.real(np.fft.ifftn(grid))))
    #four = np.abs(np.real(np.fft.ifftn(grid)))

    x = np.linspace(-scale, scale, n)
    y = np.linspace(-scale, scale, n)
    X, Y = np.meshgrid(x, y)
    Z = grid[:,:,0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    Z2 = four[:,:,0]
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.contour3D(X, Y, Z2, 50, cmap='binary')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    print('See plot')
    plt.show()