import torch
import gpytorch
import numpy.polynomial.legendre as L
import numpy as np

from legendre_function import Legendre_covar

'''
    Skapar Legendre Polynomial Kernel
'''
class LegendrePolynomial_kernel(gpytorch.kernels.Kernel):

  def __init__(self, orders, **kwargs):
        super().__init__(**kwargs)

        # Registerar coeff hyperparametern
        self.register_parameter(
            name='raw_coeff', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, len(orders)))
        )
        
        # set the parameter constraint to be positive, when nothing is specified
        coeff_constraint = gpytorch.constraints.Positive()

        # register the constraint
        self.register_constraint("raw_coeff", coeff_constraint)

        self.orders = orders

  def forward(self, x1, x2=None, **params):
    norm_x1 = torch.nn.functional.normalize(x1)
    if x2 is not None:
      norm_x2 = torch.nn.functional.normalize(x2)
    return Legendre_covar.apply(norm_x1, self.coeff, self.orders, norm_x2)

  # now set up the 'actual' parameter
  @property
  def coeff(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_coeff_constraint.transform(self.raw_coeff)

  @coeff.setter
  def coeff(self, value):
     return self._set_coeff(value)


  def _set_coeff(self, value):
    if not torch.is_tensor(value):
      value = torch.as_tensor(value).to(self.coefficients)
    self.initialize(coefficients=value)