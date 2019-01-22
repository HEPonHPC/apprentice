import numpy as np

"""
Generate random uniform arguments to arctanh.
"""


np.random.seed(100)

NPOINTS=100

# Sample in [-0.95, 0.95)
X = 1.9*np.random.random_sample((NPOINTS,)) - 0.95
Y = np.arctanh(X)

# Store as CSV
np.savetxt("pseudorapidity.csv", [(x,y) for x, y in zip(X,Y)], delimiter=',')

# Restore test
D = np.loadtxt("pseudorapidity.csv", delimiter=',')

# Plot restored data over original data just to be safe.

import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.scatter(D[:,0], D[:,1], s=1)
plt.savefig("pseudorapidity.pdf")
