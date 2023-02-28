import torch
import gpytorch
import numpy.polynomial.legendre as L
import numpy as np

from kernel import LegendrePolynomial_kernel


'''
  Skapar GP modellen
  Bestående av:
  Mean Module: Zero Mean 
  Covar Module: Legendre Kernel + RBF kernel
'''

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module1 = LegendrePolynomial_kernel(orders=(0, 2, 4, 6))
        self.covar_module2 = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module1(x) * self.covar_module2(self.dot_prod_self(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def covar(self, x, x2):
      return self.covar_module1(x, x2) * self.covar_module2(self.dot_prod_self(x), self.dot_prod_self(x2))

    def dot_prod_self(self, x):
      size_x = x.size() # batch_dim och dim 
      X = x.reshape(size_x[0], 1, size_x[1]) 
      Y = x.reshape(size_x[0], size_x[1], 1)
      product = torch.matmul(X, Y).squeeze(1)
      return product




