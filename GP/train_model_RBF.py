import torch
import gpytorch
import numpy as np
import random
#from GP_model import GPModel
#from RBF import ExactGPModel
import itertools
'''
  Fil för att träna GP'n 
'''


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def E(q,phi=90):
    # Returnerar syntetiska värdet för en viss punkt i q-space
    q= np.array(q)
    D0 = 2.5e-9
    #D0 = 10
    #D0 = 0.1
    D1 = np.diag([1, 0.1, 0.1]) * D0
    theta = np.radians(phi)
    D2 = np.diag([1*np.cos(theta)+0.1*np.sin(theta), 1*np.sin(theta)+0.1*np.cos(theta), 0.1]) * D0
    qt = np.transpose(q)
    td = 0.01
    return 0.5*(np.exp(-td*qt.dot(D1).dot(q))+np.exp(-td*qt.dot(D2).dot(q)))

def synt_data():
    scale = 1000
    n = 15
    ang = [10*i for i in range(10)]
    point = Mesh(n)
    y = []
    x = None
    for j in point:
        i = random.choice(ang)
        y.append(E(torch.mul(j, scale), i))
        if x == None:
            x = j.reshape((1,3))
        else:
            x = torch.cat((x,j.reshape((1,3))), axis=0)
    return x, torch.tensor(y)

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
  
x, y = synt_data()
for j in range(1):
    train_x = x[0:1000]
    train_y = y[0:1000]
    print(train_x.shape)
    print(train_y.shape)
    print(type(train_x))
    print(type(train_y))
    # Parameters to change before training
    load_model = False
    output_file = 'gp_model_state2.pth' # Ändra om vi vill spara tränade modellen i annan fil, oklart om man behöver skapa den själv? 
    training_iter = 1000

    # Implementera metod som fixar träningsdata här
    #train_x = torch.tensor([[0.3, 0.2], [0.3, 0.4], [0.5, 0.1]], dtype=torch.double)
    #train_y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.double)


    # Create Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood) # Kolla orders i GPModel så vi har rätt ording på Legendrepolynomen!  

    # Load previous model parameters
    if load_model:
        state_dict = torch.load('gp_model_state3.pth')
        model.load_state_dict(state_dict)

    # Use the adam optimizer and marginal log likelihood 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    # Training Loop
    for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            predictions = likelihood(model(train_x))
            # Calc loss and backprop gradients
            loss = -mll(predictions, train_y)
            # print('Loss', loss) # Uncomment for print everytime
            loss.backward()
            optimizer.step()

            # Print every 10 iterations
            if i % 10 == 0:
                print('Loss', loss, 'Iter: ', i)

    # Save model state
    torch.save(model.state_dict(), output_file)

    # Print Kernel parameters
    m = model.state_dict()
    print()
    print('Model parameters:')
    #for element in m:
    #    print(element, ':  ', m[element])
    #print('Legendre kernel coeff:', model.covar_module.coeff)
    #print('RBF kernel lengthscale: ', model.covar_module2.lengthscale)







