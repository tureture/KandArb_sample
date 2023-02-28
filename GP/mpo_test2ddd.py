import numpy as np
from os.path import join
import nibabel as nib
from dipy.data import fetch_taiwan_ntu_dsi
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
from GP_model import GPModel

torch.autograd.set_grad_enabled(False)
"""""
Kör första delen på flera kärnor samtidigt. Användbar när en dator med många kärnor används och mängden punkter är många.
"""""

def ret_data():
    files, folder = fetch_taiwan_ntu_dsi()
    nii = join(folder,'DSI203.nii.gz')
    vec = join(folder, 'DSI203.bvec')
    val = join(folder, 'DSI203.bval')

    bval, bvec = dipy.io.read_bvals_bvecs(val, vec)
    img = nib.load(nii)
    data = img.get_data()
    dat = []
    for i in range(len(bval)):
        mag = bval[i]
        grad = bvec[i]
        dat.append([mag*j for j in grad])
    return torch.tensor(dat)



def E(q,phi=90):
    # Returnerar syntetiska värdet för en viss punkt i q-space
    q= np.array(q)
    D0 = 2.5e-3
    #D0 = 10
    #D0 = 0.1
    D1 = np.diag([1, 0.1, 0.1]) * D0
    theta = np.radians(phi)
    D2 = np.diag([1*np.cos(theta)+0.1*np.sin(theta), 1*np.sin(theta)+0.1*np.cos(theta), 0.1]) * D0
    qt = np.transpose(q)
    td = 0.02
    return 0.5*(np.exp(-td*qt.dot(D1).dot(q))+np.exp(-td*qt.dot(D2).dot(q)))


def sphere_mesh(n, reduced=False):
    r = np.linspace(0,1,n)
    if reduced:
        phi = np.linspace(0,np.pi,(n//2)+1)
    else:
        phi = np.linspace(0,2*np.pi,n)
    theta = np.linspace(0,np.pi,(n//2)+1)
    mesh = []
    for i in r:
        for j in theta:
            for k in phi:
                mesh.append((i*np.sin(j)*np.cos(k),i*np.sin(j)*np.sin(k),i*np.cos(j)))
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
    z = np.linspace(-space,space,n)
    return torch.tensor(list(itertools.product(x, y, z)))


def diff_val(y,S,A,Kern):
    y_re = y.reshape(1,-1)
    total_sensor = torch.cat((A,y_re), 0)
    #Se = S[~S.unsqueeze(1).eq(total_sensor).all(-1).any(-1)]
    Se = S
    SS = Kern.covar(Se,Se).evaluate() + torch.diag(torch.abs(torch.randn(Se.shape[0])/(100))) # Lägger till brus för att göra den inverterbar
    vy = Kern.covar(y_re,y_re).evaluate() 
    AA = Kern.covar(A,A).evaluate() + torch.diag(torch.abs(torch.randn(A.shape[0])/(100)))
    


    #yA = 't'    # Ful lösning men funkar
    #for i in A:
    #    if yA == 't':
    #        yA = Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)
    #    else:
    #        yA = torch.cat((yA, Kern(torch.cat((y_re, i.reshape(1,-1)), 0)).evaluate()[0][1].reshape(1,-1)),0)
    #yA = Kern.forward(y_re.float(), A.float())
    yA = Kern.covar(y_re,A).evaluate()
    
    yA = yA.reshape(-1,1)
    yS = Kern.covar(y_re,Se).evaluate()
    yS = yS.reshape(-1,1)
    AAinv = np.linalg.solve(AA.numpy(), yA.numpy())
    SSinv = np.linalg.solve(SS.numpy(), yS.numpy())
    yAt = torch.t(yA)
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

def initi(y,S,A,Kern,q):
    q.append([diff_val(y,S,A,Kern).detach(), y])

if __name__ == '__main__':
    # Skapar default kernel, ska ändras till kernel som beskrivs i Sjölund vid senare tillfälle.
    Kern = GPModel([],[],gpytorch.likelihoods.GaussianLikelihood())
    Kern.load_state_dict(torch.load('gp_model_state2.pth'))
    #Kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # Kern = LegendrePolynomial_simple()
    #Kern = GPModel([],[],gpytorch.likelihoods.GaussianLikelihood())
    n = 15 # 7 140s 11 1070
    #S = sphere_mesh(n)
    Sred = sphere_mesh(n,True)
    print(Sred.shape)
    #Sred = Mesh(n, True)
    A = torch.tensor([[0, 0, 0]], dtype=torch.float64) # Skapar en utgångspunkt i 0,0,0 att arbeta från.
    cand = Sred[~Sred.unsqueeze(1).eq(A).all(-1).any(-1)] # Skapar delmängden S\A, alltså platser för nästa sensor
    S = Mesh(n)
    #S = sphere_mesh(n)
    q = []
    manager = mp.Manager()
    lis = manager.list()
    #AA = Kern.forward(A).evaluate() # Skapar vår AA covariance matris
    # AA = torch.tensor(Kern.K(A,A))
    #AAinv = torch.inverse(AA) # Inverterar den.
    #any = AAinv.detach()
    inputs = []
    tic = time.perf_counter()
    print(f'Starting initial evaluation for approximatley {(n**2)/2} points')
    for i in cand:
        inputs.append((i,S,A,Kern,lis))
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
    num_sensor = 205
    print(f'Starting placement of {num_sensor} sensors')
    for k in range(num_sensor):
        #AA = Kern.covar(A,A).evaluate() # Skapar vår AA covariance matris
        # AA = Kern.K(A,A)
        #AAinv = torch.inverse(AA +torch.tensor(np.random.normal(0,0.0001,AA.shape))) # Inverterar den.
        run = True
        current = set()
        while run == True:
            diff_old, ind, y = heapq.heappop(q)
            if y in current:
                run = False
            diff = diff_val(y,S,A,Kern)
            if diff > -diff_old:
                print('FUCK THIS')
                print(f'Förra:{-diff_old, diff}')
            if run == False:
                besty = y
            else:
                heapq.heappush(q, (-diff, ind, y))
                current.add(y)
        A = torch.cat((A,besty.reshape(1,-1)), 0)
        A = torch.cat((A,-besty.reshape(1,-1)), 0)
        if last_diff > diff:
            print('Monotonic behavior')
            print(f'Current diff: {diff}, Last diff: {last_diff}')
        last_diff = diff
        if diff < 0:
            print('FML') # 140s
        gc.collect()
        
        #inv = torch.tensor((-besty[0], besty[1]))
        #A = torch.cat((A,inv.reshape(1,-1)), 0)
        print(f'Sensor {k+1} placed')
    toc2 = time.perf_counter()
    print(f'Finished sensor placement!\nTime for section: {toc2-toc}')
    print(A)
    # Add symmetric parts
    torch.save(A,'MI_placement.pt')
    scale = 1000
    grid_facit = np.zeros((n,n,n))
    
    for sensor in S:
        x = int(round(float((sensor[0]+1)*(n-1)/2)))
        y = int(round(float((sensor[1]+1)*(n-1)/2)))
        z = int(round(float((sensor[2]+1)*(n-1)/2)))
        val = E(sensor*scale,90)
        if grid_facit[x,y,z] != 0:
            print('Duplicate detected')
            print(sensor)
        grid_facit[x,y,z]= val

    four = np.fft.ifftshift(np.abs(np.real(np.fft.ifftn(grid_facit))))
    #four = np.abs(np.real(np.fft.ifftn(grid)))
    grid2 = np.zeros((n,n,n))
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
        points = np.vstack((points,[[-1,i,-1]]))
        points = np.vstack((points,[[-1,i,1]]))
        points = np.vstack((points,[[1,i,-1]]))
        points = np.vstack((points,[[1,i,1]]))
        points = np.vstack((points,[[i,-1,-1]]))
        points = np.vstack((points,[[i,-1,1]]))
        points = np.vstack((points,[[i,1,1]]))
        points = np.vstack((points,[[i,1,-1]]))
        points = np.vstack((points,[[1,1,i]]))
        points = np.vstack((points,[[1,-1,i]]))
        points = np.vstack((points,[[-1,1,i]]))
        points = np.vstack((points,[[-1,-1,i]]))
    datapoint = datapoint + (12*n*[0])
    datapoint = np.array(datapoint)
    print(points.shape)
    print(datapoint.shape)
    xi = Mesh(n)
    gridcord = griddata(points,datapoint,xi)
    grid = np.zeros((n,n,n))


    for e, i in enumerate(xi):
        x = int(round(float((i[0]+1)*(n-1)/2)))
        y = int(round(float((i[1]+1)*(n-1)/2)))
        z = int(round(float((i[2]+1)*(n-1)/2)))
        grid[x,y,z] = gridcord[e]
    
    #print(grid)
    four2 = np.fft.ifftshift(np.abs(np.real(np.fft.ifftn(grid))))
    #x = map(abs,(four-four2)*(1))
    print(sum(sum(sum(np.absolute(four-four2)))))
    x = np.linspace(-scale, scale, n)
    y = np.linspace(-scale, scale, n)

    X, Y = np.meshgrid(x, y)
    Z = grid_facit[:,:,0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    Z3 = grid[:,:,0]
    fig3 = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.contour3D(X, Y, Z3, 50, cmap='binary')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    Z2 = four2[:,:,0]
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.contour3D(X, Y, Z2, 50, cmap='binary')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    print('See plot')
    plt.show()