import autograd.numpy as np
from autograd import grad
import torch

# a named Python function
def g(a, b, c, d, e):
    x = torch.tensor([a, b, c, d, e], requires_grad=True)

    return torch.median(x)


dgdw1 = grad(g, 0)
print(dgdw1)

print(dgdw1(17997,2,3,4,5))