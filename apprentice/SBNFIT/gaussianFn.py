import argparse
import numpy as np
import os,sys
import h5py
import json

def gaussianFn(c,x,mu,cov,a):
    # sd = [cov[0,0],cov[1,1]]
    # e1 = 0.
    # for xno,xd in enumerate(x):
    #     e1 += ((x[xno]-mu[xno])**2)/(2 * sd[xno] ** 2)
    # e1 = np.exp(-1*e1)
    #
    # return c+(a * e1)



    diff = x - mu
    cov2 = np.matmul(cov.transpose(),cov)
    e2 = -0.5 * (
        np.matmul(
            np.matmul(
                diff.transpose(),
                np.linalg.inv(cov2)),
            diff
        )
    )
    e2 = np.exp(e2)

    # if x[0]>mu[0]-sd[0] and x[0]<mu[0]+sd[0] and \
    #     x[1]>mu[1]-sd[1] and x[1]<mu[1]+sd[1]:
    #     print(x)
    #     print(e1, e2)
    #     print(c + (a * e1),(c + (a * e2)))


    return c + (a * e2)


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

    signalds = {"binids":[],"a":[],"mu":[],"cov":[],"c":[],"Xall":Xall,
                "npointsperdim":npointsperdim,"nbin":nbin}

    for bno, bin in enumerate(binids):
        Yall = data[:, bno]
        signalds["binids"].append(bin)
        signalds["c"].append(np.max(Yall))
        minindex = int(np.argmin(Yall))
        signalds["mu"].append(Xall[minindex])
        signalds["a"].append(-1*(np.max(Yall) - np.min(Yall)))
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
        signalds['cov'].append([[sd[0],0],[0,sd[1]]])

    return signalds

"""
python gaussianFn.py -o ../../../log/SBNFIT/plots -i ../../../log/SBNFIT/comparespectrum_mpi_deg2.h5
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian Mock data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default="/tmp",
                        help="Output directory")
    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    parser.add_argument("-d", "--dimension", dest="DIM", type=str, default=None,
                        help="Dimension")
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
    cov_arr = signalds["cov"]
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
                                              np.array(cov_arr[bno]),
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





