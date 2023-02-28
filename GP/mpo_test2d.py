import numpy as np
from os.path import join
import nibabel as nib
import scipy
import torch
import gpytorch
import dipy
import heapq
import itertools
import cProfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import heapq
import time
import gc
from scipy.interpolate import griddata


torch.autograd.set_grad_enabled(False)
"""""
Kör första delen på flera kärnor samtidigt. Användbar när en dator med många kärnor används och mängden punkter är många.
"""""
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def E(q,phi=90):
    # Returnerar syntetiska värdet för en viss punkt i q-space
    q= np.array(q)
    D0 = 2.5e-9
    #D0 = 10
    #D0 = 0.1
    D1 = np.diag([1, 0.1]) * D0
    theta = np.radians(phi)
    D2 = np.diag([1*np.cos(theta)+0.1*np.sin(theta), 1*np.sin(theta)+0.1*np.cos(theta)]) * D0
    qt = np.transpose(q)
    td = 0.075
    return 0.5*(np.exp(-td*qt.dot(D1).dot(q))+np.exp(-td*qt.dot(D2).dot(q)))


def linear_solver(A,b):
    L = np.linalg.cholesky(A)
    y = lin_sol(L,b)
    x = lin_sol(np.transpose(L),y)
    return x

def synt_data():
    n = 10
    ang = [10*i for i in range(10)]
    point = Mesh(n)
    y = []
    x = []
    for i in ang:
        for j in point:
            y.append(E(j, i))
            x.append(j)
    return x, torch.tensor(y)


def lin_sol(L_org,b_org):
    n = len(L_org[0])
    flipped = 0
    if np.count_nonzero(L_org[0]) != 1:
        L = np.flip(L_org,0)
        b = np.flip(b_org,0)
        flipped = 1
    else:
        L = L_org
        b = b_org
    x = np.zeros(n)
    for i in range(n):
        current  = L[i].copy()
        current = current/(current[i])
        x[i] = b[i] - np.dot(current[:i],x[:i])
    x = x.reshape((-1,1))
    if flipped == 1:
        return np.flip(x)
    else:
        return x

def sphere_mesh(n, reduced=False):
    r = np.linspace(0,1,n)
    if reduced:
        theta = np.linspace(0,np.pi/2,(n//2)+1)
    else:
        theta = np.linspace(0,np.pi,n)
    mesh = []
    for i in r:
        for j in theta:
            mesh.append((i*np.cos(j),i*np.sin(j)))
    return torch.tensor(mesh)

def Mesh(n, reduced=False):
    # Skapar meshen över de q-space punkter vi vill testa
    # Eftersom E(q) = E(-q) så behöver vi bara kolla på halva rummet. Däran n//2 på x.
    # Borde öka prestandan med x8. Kan optimeras yttligare genom att ha varierande stegstorlek.
    space = 1
    if reduced==True:
        x = np.linspace(0,space,(n//2)+1) 
    else:
        x = np.linspace(-space,space,n)
    y = np.linspace(-space,space,n)
    return torch.tensor(list(itertools.product(x, y)))

def own_ker(a,b):
    # Ska testa om bättre för fallet med bara två värden än forward
    dist = (a - b).pow(2).sum().sqrt()
    return np.exp((dist**2)*0.5)

def diff_val(y,S,A,AAinv,Kern):
    y_re = y.reshape(1,-1)
    total_sensor = torch.cat((A,y_re), 0)
    Se = S[~S.unsqueeze(1).eq(total_sensor).all(-1).any(-1)]
    SS = Kern(Se).evaluate() + torch.diag(torch.abs(torch.randn(Se.shape[0])/(100))) # Lägger till brus för att göra den inverterbar
    vy = new = own_ker(y_re,y_re)
    AA = Kern(A).evaluate() + torch.diag(torch.abs(torch.randn(A.shape[0])/(100)))
    
    #SS = Kern.K(Se,Se)
    #vy = Kern.K(y_re,y_re)
    # SSinv = torch.inverse(SS)
    

    # Skapar yA covariance matrisen
    #yA = Kern.K(y_re,A)
    #yAt = yA.t()
    #yS = Kern.K(y_re,Se)
    #ySt = yA.t()

    yA = 't'    # Ful lösning men funkar
    for i in A:
        if yA == 't':
            yA = Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)
        else:
            yA = torch.cat((yA, Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)),0)
    #yA = Kern.forward(y_re.float(), A.float())
    yAt = torch.t(yA)
    yS = Kern.forward(y_re,Se)
    yS = yS.reshape(-1,1)
    AAinv = np.linalg.solve(AA.numpy(), yA.numpy())
    SSinv = np.linalg.solve(SS.numpy(), yS.numpy())

    ySt = torch.t(yS)
    tal = (vy-torch.matmul(yAt.float(),torch.tensor(AAinv.copy()).float()))
    if tal < 0:
        print(f'Blame tal {tal}')
    den = (vy-torch.matmul(ySt.float(),torch.tensor(SSinv.copy()).float()))
    if den < 0:
        print(f'Blame den {den}')
    diff = tal/den
    del yS, ySt, Se, SS, SSinv, vy, yA, yAt
    return diff

def initi(y,S,A,AAinv,Kern,q):
    q.append([diff_val(y,S,A,AAinv,Kern).detach(), y])

if __name__ == '__main__':
    # Skapar default kernel, ska ändras till kernel som beskrivs i Sjölund vid senare tillfälle.
    Kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # Kern = LegendrePolynomial_simple()
    n = 11
    S = sphere_mesh(n)
    Sred = sphere_mesh(n,True)
    A = torch.tensor([[0, 0]]) # Skapar en utgångspunkt i 0,0,0 att arbeta från.
    cand = Sred[~Sred.unsqueeze(1).eq(A).all(-1).any(-1)] # Skapar delmängden S\A, alltså platser för nästa sensor
    S = Mesh(n)
    q = []
    manager = mp.Manager()
    lis = manager.list()
    AA = Kern(A).evaluate() # Skapar vår AA covariance matris
    # AA = torch.tensor(Kern.K(A,A))
    AAinv = torch.inverse(AA) # Inverterar den.
    any = AAinv.detach()
    inputs = []
    tic = time.perf_counter()
    print(f'Starting initial evaluation for approximatley {(n**2)/2} points')
    for i in cand:
        inputs.append((i,S,A,any,Kern,lis))
    p = mp.Pool(mp.cpu_count())
    p.starmap(initi,inputs)
    p.close()
    p.join()
    last_diff = 0
    for e, i in enumerate(lis):
        heapq.heappush(q, (-i[0], e, i[1]))
    toc = time.perf_counter()
    print(f'Initial evaluation finished!\nTime: {toc-tic}')
    # Loop som ger oss de bästa sensor placeringarna.
    num_sensor = 30
    print(f'Starting placement of {num_sensor} sensors')
    for k in range(num_sensor):
        AA = Kern(A).evaluate() # Skapar vår AA covariance matris
        # AA = Kern.K(A,A)
        AAinv = torch.inverse(AA +torch.tensor(np.random.normal(0,0.0001,AA.shape))) # Inverterar den.
        run = True
        current = set()
        while run == True:
            diff_old, ind, y = heapq.heappop(q)
            if y in current:
                run = False
            diff = diff_val(y,S,A,AAinv,Kern)
            if run == False:
                besty = y
            else:
                heapq.heappush(q, (-diff, ind, y))
                current.add(y)
        A = torch.cat((A,besty.reshape(1,-1)), 0)
        A = torch.cat((A,-besty.reshape(1,-1)), 0)
        if last_diff > diff:
            print('Non-monotonic behavior')
            print(f'Current diff: {diff}, Last diff: {last_diff}')
        last_diff = diff
        if diff < 0:
            print('FML')
        gc.collect()
        
        #inv = torch.tensor((-besty[0], besty[1]))
        #A = torch.cat((A,inv.reshape(1,-1)), 0)
        print(f'Sensor {k+1} placed')
    toc2 = time.perf_counter()
    print(f'Finished sensor placement!\nTime for section: {toc2-toc}')
    print(A)
    # Add symmetric parts

    scale = 1000
    grid = np.zeros((n,n))
    
    for sensor in A:
        x = int(round(float((sensor[0]+1)*(n-1)/2)))
        y = int(round(float((sensor[1]+1)*(n-1)/2)))
        val = E(sensor*scale,90)
        if grid[x,y] != 0:
            print('Duplicate detected')
            print(sensor)
        grid[x,y]= val

    four = np.fft.ifftshift(np.abs(np.real(np.fft.ifft2(grid))))
    #four = np.abs(np.real(np.fft.ifftn(grid)))
    grid2 = np.zeros((n,n))
    points = np.array(A)
    datapoint = []
    for sensor in A:
        #x = int(round(float((sensor[0]+1)*(n-1)/2)))
        #y = int(round(float((sensor[1]+1)*(n-1)/2)))
        val = E(sensor*scale,90)
        datapoint.append(val)
        #if grid2[x,y] != 0:
        #    print('Duplicate detected')
        #    print(sensor)
        #grid2[x,y]= val
    for i in np.linspace(-1,1,n):
        points = np.vstack((points,[[-10,i]]))
        points = np.vstack((points,[[10,i]]))
        points = np.vstack((points,[[i,-10]]))
        points = np.vstack((points,[[i,10]]))
    datapoint = datapoint + (4*n*[0])
    datapoint = np.array(datapoint)
    print(points.shape)
    print(datapoint.shape)
    xi = np.array(S)
    gridcord = griddata(points,datapoint,xi)
    grid = np.zeros((n,n))

    for e, i in enumerate(xi):
        x = int(round(float((i[0]+1)*(n-1)/2)))
        y = int(round(float((i[1]+1)*(n-1)/2)))
        grid[x,y] = gridcord[e]
        
    print(grid)
    four2 = np.fft.ifftshift(np.abs(np.real(np.fft.ifft2(grid))))
    #x = map(abs,(four-four2)*(1))
    print(sum(sum(np.absolute(four-four2))))
    x = np.linspace(-scale, scale, n)
    y = np.linspace(-scale, scale, n)
    X, Y = np.meshgrid(x, y)
    Z = grid
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    Z2 = four2
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.contour3D(X, Y, Z2, 50, cmap='binary')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    print('See plot')
    plt.show()