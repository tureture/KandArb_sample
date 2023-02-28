import torch
import numpy.polynomial.legendre as L
import numpy as np

# Skapar Legendre Polynomial kernels kovarians funktion.
# Implementerad som torch.function
# Innehåller en forward och backward metod

# Tar input torch.tensorer
# Omvandlar till numpy inuti
# Returnerar torch.tensorer

class Legendre_covar(torch.autograd.Function):

    # Beräknar kovariansmatrisen 
    def forward(ctx, x1, coeff, orders, x2 = None):
        V, c = Legendre_covar._getVandermondeMatrixAndCoefficients(x1, X2=x2, orders=orders, coeff=coeff)
        
        ctx.V = V
        ctx.orders = orders

        output = np.dot(V,c)
        return torch.from_numpy(output)

    def _getVandermondeMatrixAndCoefficients(X1, coeff, orders, X2 = None):
        """
        Compute and cache the pseudo-Vandermonde matrix V and inflate the
        coefficients into a 1-D array c of length n + 1. The Legendre
        polynomial can then be evaluated as np.dot(V, c).
        """
        if X2 is None:
            dot_prod = np.dot(X1.numpy(), X1.numpy().T)
        else:
            dot_prod = np.dot(X1.numpy(), X2.numpy().T)
        highestOrder = np.max(orders)
        V = L.legvander(dot_prod, highestOrder)
        c = np.zeros(highestOrder + 1)
        c[np.array(orders)] = coeff 
        return V, c

    # Beräknar gradienten för coeff
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        V = ctx.V
        orders = ctx.orders
        grad_output = grad_output.numpy()

        ''' 
        # Orginal update_gradients 
        V = V[:, :, self.orders]
        dL_dCoefficients = dL_dK[:, :, np.newaxis] * V
        self.coefficients.gradient = np.sum(dL_dCoefficients, axis=(0, 1))
        '''

        V = V[:, :, orders]
        grad_input = grad_output[:, :, np.newaxis] * V 
        out = np.sum(grad_input, axis=(0, 1))
        out = torch.from_numpy(out)
        
        return None, torch.unsqueeze(out, 0), None, None
        