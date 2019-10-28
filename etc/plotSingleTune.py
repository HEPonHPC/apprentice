import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("ggplot")

import json, sys
with open(sys.argv[1]) as f:
    D = json.load(f)

V = [v[0] for k, v in D.items()]
plt.hist(V,bins=100)
plt.savefig("singletunechi2dist.pdf")

plt.clf()

import numpy as np
P = np.array([v[1:] for k, v in D.items()])

dim=len(P[0])

for i in range(dim):
    for j in range(dim):
        if j>=i:continue
        plt.clf()
        plt.scatter(P[:,i], P[:,j], c=V)
        plt.xlabel("p {}".format(i))
        plt.ylabel("p {}".format(j))
        bar = plt.colorbar()
        bar.set_label("gof")
        plt.savefig("singletunepcorr_{}_{}.pdf".format(i,j))



