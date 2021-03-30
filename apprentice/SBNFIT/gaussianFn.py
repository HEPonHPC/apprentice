import argparse
import numpy as np
import os,sys
import h5py
import json

def gaussianFn(c,x,mu,sd,a):
    e = 0.
    for xno,xd in enumerate(x):
        e += ((x[xno]-mu[xno])**2)/(2 * sd[xno] ** 2)
    e = -np.exp(-1*e)

    return c+(a * e)

def getSignalData(infile):
    f = h5py.File(infile, "r")
    data = np.array(f.get('colspec'))
    f.close()
    nbin = np.shape(data)[1]
    npoints = np.shape(data)[0]
    binids = ["Bin{}".format(i) for i in range(nbin)]
    npointsperdim = int(np.sqrt(npoints))
    Xall = []
    dim = 2
    for i in range(1, npointsperdim + 1):
        for j in range(1, npointsperdim + 1):
            Xall.append([i, j])

    signalds = {"binids":[],"a":[],"mu":[],"sd":[],"c":[],"Xall":Xall,
                "npointsperdim":npointsperdim,"nbin":nbin}

    for bno, bin in enumerate(binids):
        Yall = data[:, bno]
        signalds["binids"].append(bin)
        signalds["c"].append(np.max(Yall))
        minindex = int(np.argmin(Yall))
        signalds["mu"].append(Xall[minindex])
        signalds["a"].append(np.max(Yall) - np.min(Yall))
        halfl10 = (np.log10(np.max(Yall)) + np.log10(np.min(Yall)))/2
        half = 10**halfl10
        # half = (np.max(Yall) +np.min(Yall))/2
        sd = [0] * dim
        for d in range(dim):
            for x in range(1,Xall[minindex][d]):
                xx = [kk for kk in Xall[minindex]]
                xx[d] = x
                index = Xall.index(xx)
                # print(Yall[index])
                if Yall[index] < half:
                    sd[d] = Xall[minindex][d] - x
                    # print(Xall[minindex][d],x)
                    break
        signalds['sd'].append(sd)
    return signalds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian Mock data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default="/tmp",
                        help="Output directory")
    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    args = parser.parse_args()

    signalvals = None
    valtype = "GaussianFnSignal"
    if args.INFILE is not None:
        if "json" in args.INFILE:
            with open(args.INFILE,"r") as f:
                signalds = json.load(f)
        else:
            signalds = getSignalData(args.INFILE)
            with open(os.path.join(args.OUTDIR,"{}.json".format(valtype)), "w") as f:
                json.dump(signalds,f,indent=4)
    else:
        raise Exception("INFILE needs to be specified")

    nbin = signalds["nbin"]
    npointsperdim = signalds["npointsperdim"]
    Xall = signalds["Xall"]
    # Height
    a_arr = signalds["a"]
    # Mean
    mu_arr = signalds["mu"]
    # Standard Deviation
    sd_arr = signalds["sd"]
    # Shift
    c_arr = signalds["c"]
    binids = signalds["binids"]

    import seaborn as sns
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(args.OUTDIR, valtype), exist_ok=True)
    # Parameter
    signalvals = {}
    for bno,bin in enumerate(binids):
        signalvals[bin] = []
        for x in Xall:
            signalvals[bin].append(gaussianFn(c_arr[bno],
                                              np.array(x),
                                              np.array(mu_arr[bno]),
                                              np.array(sd_arr[bno]),
                                              a_arr[bno]))
        # print(signalvals[bin])
        fig, ax = plt.subplots(figsize=(12, 7))
        title = "{} {} Heat Map".format(bin, "Gaussian function signal")
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])

        ax.set_xticks([])
        ax.set_yticks([])

        ax.axis('off')
        # print(minv)
        # print(maxv)
        vvv = np.log10(signalvals[bin])
        # vvv = np.array(signalvals[bin])
        sns.heatmap(vvv.reshape(npointsperdim, npointsperdim),
                    fmt="", cmap="RdYlGn", linewidths=0.3, ax=ax
                    # ,vmin=minv,vmax=maxv
                    )
        # plt.show()
        plt.savefig(os.path.join(args.OUTDIR, valtype, "{}_{}.pdf".format(bin, valtype)))
        plt.close()


    # print(signalvals[binids[0]])





