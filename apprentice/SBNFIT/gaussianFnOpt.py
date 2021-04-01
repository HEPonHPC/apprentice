import argparse
import numpy as np
import os,sys
import h5py
import json
from scipy.optimize import minimize
dim = 2

def objective(x,Xall,Yall):
    c = x[0]
    a = x[1]
    mu = np.array([x[2],x[3]])
    cov = np.array([[x[4],x[5]],[x[6],x[7]]])

    lsqterms = [
        (gaussianFn(c, Xall[pno], mu, cov, a) - Yall[pno]) ** 2
        for pno in range(len(Xall))
    ]
    return sum(lsqterms)


def gaussianFn(c,x,mu,cov,a):
    diff = x - mu
    cov2 = np.matmul(cov.transpose(), cov)
    e = -0.5 * (
        np.matmul(
            np.matmul(
                diff.transpose(),
                np.linalg.inv(cov2)),
            diff
        )
    )
    e = np.exp(e)

    return c + (a * e)


def optimize(infile):
    f = h5py.File(infile, "r")
    data = np.array(f.get('colspec'))
    f.close()
    nbin = np.shape(data)[1]
    npoints = np.shape(data)[0]
    binids = ["Bin{}".format(i) for i in range(nbin)]
    npointsperdim = int(np.sqrt(npoints))
    Xall = []
    for i in range(1, npointsperdim + 1):
        for j in range(1, npointsperdim + 1):
            Xall.append([i, j])
    XallMain = Xall
    signalds = {"binids": [], "a": [], "mu": [], "cov": [], "c": [], "Xall": Xall,
                "npointsperdim": npointsperdim, "nbin": nbin}

    bounds = [(0,np.infty),(-np.infty,0),(1,26),(1,26),
              (-np.infty, np.infty),(-np.infty, np.infty),
              (-np.infty, np.infty),(-np.infty, np.infty)]
    for bno, bin in enumerate(binids):
        Yall = data[:, bno]
        signalds["binids"].append(bin)
        x0 = np.array([1.,1.,26,26,1.,0.,0.,1.])
        ret = minimize(fun=objective, x0=x0,
                       args=(Xall,Yall), bounds=bounds,
                       method='L-BFGS-B')
        var = ret.get('x')
        signalds['c'].append(var[0])
        signalds['a'].append(var[1])
        mu = [var[i] for i in range(2,4)]
        signalds['mu'].append(mu)
        cov = [[var[i]for i in range(4,6)],[var[i]for i in range(6,8)]]
        signalds['cov'].append(cov)
        sys.stdout.write("Done with bin {} out of {} bins  \r".format(bno+1,len(binids)))
    return signalds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian Mock data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default="/tmp",
                        help="Output directory")
    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    args = parser.parse_args()

    signalds = optimize(args.INFILE)
    valtype = "GaussianFnSignal_Optimized"
    with open(os.path.join(args.OUTDIR, "{}.json".format(valtype)), "w") as f:
        json.dump(signalds, f, indent=4)
