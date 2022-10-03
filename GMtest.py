import torch
from geom_median.torch import compute_geometric_median
import time
import math
import random
num = 0
denom = 0
t0 = time.time()
for i in range(2 ** 17):
    x = random.randint(1, 4)
    # num = num + 1 / math.sqrt((random.randint(1, 4) + 1))
    #denom = denom + 1 / math.sqrt((random.randint(1, 4) + 1))
# x = num/denom
t1 = time.time()
print(t1-t0)