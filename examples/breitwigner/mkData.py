import numpy as np

"""
Breit-Wigner
"""


def gamma(M, G):
    return np.sqrt(M*M * (M*M + G*G))

def kprop(M, G):
    y = gamma(M,G)
    return (2*np.sqrt(2) * M * G * y ) / (np.pi * np.sqrt(M*M + y))

def BW(E, M=91.2, G=10):
    return kprop(M, G) / ( (E*E - M*M)*(E*E - M*M) + M*M*G )


np.random.seed(100)


NPOINTS=1000
E = 80 +  20 * np.random.random_sample((NPOINTS,))
G =  5 +   5 * np.random.random_sample((NPOINTS,))
M = 90 +   3 * np.random.random_sample((NPOINTS,))

# 1D test plot
T = BW(E)
import matplotlib.pyplot as plt
plt.scatter(E, T)
plt.savefig("breitwigner.pdf")

# Store as CSV
np.savetxt("breitwigner.csv", [(e,m,g, BW(e,m,g)) for e,m,g in zip(E,M,G)], delimiter=',')
