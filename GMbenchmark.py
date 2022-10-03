import torch
from geom_median.torch import compute_geometric_median
import time
import matplotlib.pyplot as plt
import matplotlib


n = 500  # Number of vectors
d = 1  # dimensionality of each vector
points = [torch.rand(1) for _ in range(n)]   # list of n tensors of shape (d,)
points_tensor = torch.Tensor(points)
# The shape of each tensor is the same and can be arbitrary (not necessarily 1-dimensional)

x_ax = []
means = []
medians = []
geom_medians = []

for i in range(0, 18, 1):
    n = 1 * (2 ** i)  # Number of vectors
    points = [torch.rand(1) for _ in range(n)]   # list of n tensors of shape (1,)
    points_tensor = torch.Tensor(points)
    t0 = time.time()    
    for _ in range(100):
        torch.mean(points_tensor)
    t1 = time.time()
    for _ in range(100):
        torch.median(points_tensor)
    t2 = time.time()

    x_ax.append(n)
    means.append((t1-t0) / 100)
    medians.append((t2-t1) / 100)

print(means)
print(medians)

matplotlib.rcParams.update({'font.size': 20})
plt.title('Wall Clock Time for 100 Trial Average of Mean/Median')
plt.plot(x_ax, means, label='Mean')
plt.plot(x_ax, medians, label='Median')


plt.xscale('log',base=2) 
plt.xlabel("Number of Values")
plt.ylabel("Wall Clock Time")
matplotlib.rc('legend', fontsize=20)
plt.legend()
plt.show()