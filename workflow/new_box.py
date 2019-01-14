#!/usr/bin/env python

import json
import apprentice

import numpy as np

if __name__ == "__main__":

    import sys
    M=json.load(open(sys.argv[1]))
    X = np.array(M["x"])
    Y = np.array(M["fun"])

    S=apprentice.Scaler(sys.argv[2])

    # Trivial new box based on min/max of results
    M_min = np.min(X, axis=0)
    M_max = np.max(X, axis=0)

    # Extra width
    allowance=0.25
    #
    new_box = []
    # Figure out if we hit the wall
    for num, (r, l) in enumerate(zip(np.isclose(S.sbox[:,0], S(M_min, unscale=True), rtol=1e-2), np.isclose(S.sbox[:,1], S(M_max, unscale=True), rtol=1e-2) )):
        # print(num, r,l)
        dist = M_max[num] - M_min[num]
        if not r and not l: new_box.append( [ M_min[num] - allowance*dist, M_max[num] + allowance*dist  ])
        elif r and not l:  new_box.append( [ M_min[num] - dist, M_max[num] + allowance*dist  ])
        elif l and not r:  new_box.append( [ M_min[num] - allowance*dist, M_max[num] + dist  ])
        else: print("This should never happen")


    mkPlots=True

    if mkPlots:

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(S.dim-1, S.dim-1,  figsize=(25,25))

        # [row, column], top left is [0,0]
        for col in range(S.dim-1):
            for row in range(S.dim-1):
                if col <= row:
                    axes[row, col].scatter( X[:,col], X[:,row+1], s=20, c=Y)
                else:
                    fig.delaxes(axes[row, col])

                axes[row, col].plot( X[-1][col], X[-1][row+1], "rx")
                axes[row, col].axvline(S.box[col][0], color="r", linestyle="--")
                axes[row, col].axvline(S.box[col][1], color="r", linestyle="--")
                axes[row, col].axhline(S.box[row+1][0], color="r", linestyle="--")
                axes[row, col].axhline(S.box[row+1][1], color="r", linestyle="--")
                axes[row, col].set_xlabel("$x_%i$"%col)
                axes[row, col].set_ylabel("$x_%i$"%(row+1))
        fig.savefig("oldbox.pdf")
        plt.clf()

        fig, axes = plt.subplots(S.dim-1, S.dim-1,  figsize=(25,25))
        for col in range(S.dim-1):
            for row in range(S.dim-1):
                if col <= row:
                    axes[row, col].scatter( X[:,col], X[:,row+1], s=20, c=Y)
                else:
                    fig.delaxes(axes[row, col])

                axes[row, col].plot( X[-1][col], X[-1][row+1], "rx")
                axes[row, col].set_xlabel("$x_%i$"%col)
                axes[row, col].axvline(new_box[col][0],   color="b", linestyle="--")
                axes[row, col].axvline(new_box[col][1],   color="b", linestyle="--")
                axes[row, col].axhline(new_box[row+1][0], color="b", linestyle="--")
                axes[row, col].axhline(new_box[row+1][1], color="b", linestyle="--")
                axes[row, col].set_ylabel("$x_%i$"%(row+1))
        fig.savefig("newbox.pdf")

    import json
    with open(sys.argv[3], "w") as f:
        json.dump({"newbox": new_box}, f)
