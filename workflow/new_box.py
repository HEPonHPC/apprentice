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

    # from IPython import embed
    # embed()
    print(S.box)

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
    fig.savefig("newbox.pdf")
