#!/usr/bin/env python

import json
import apprentice
import numpy as np


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    # Fix size, sometimes there is spillover
    # TODO: replace with while if problem persists
    if len(out)>num:
        out[-2].extend(out[-1])
        out=out[0:-1]

    if len(out)!=num:
        raise Exception("something went wrong in chunkIt, the target size differs from the actual size")

    return out



def readApprox(fname):
    with open(fname) as f: rd = json.load(f)
    binids = sorted(rd.keys())
    RA = [apprentice.RationalApproximation(initDict=rd[b]) for b in binids]
    return binids, RA

def readExpData(fname, binids):
    with open(fname) as f: dd = json.load(f)
    Y = np.array([dd[b][0] for b in binids])
    E = np.array([dd[b][1] for b in binids])
    return Y, E

def mkCov(yerrs):
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

if __name__ == "__main__":
    import sys
    from scipy import optimize
    import time

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    np.random.seed(rank)



    # At the moment, the object selection for the chi2 minimisation is
    # simply everything that is in the approximation file
    if rank ==0:
        binids, RA = readApprox(sys.argv[1])
        print("[{}] len bids: {}".format(rank, len(binids)))

        # S1_hnames=[
                 # # "/ATLAS_2010_S8918562/d03-x01-y01", # 50 bins from -2.5 to 2.5
                 # # "/ATLAS_2010_S8918562/d10-x01-y01",  # 36 bins from 0.5 to 50
                # # "/ATLAS_2010_S8918562/d17-x01-y01",  # 39 bins from 0.5 to 120.5
                # # "/ATLAS_2010_S8918562/d23-x01-y01"   # 38 bins from 0.5 to 99.5
                # ]
        # S2_hnames=[
                 # "/ATLAS_2010_S8918562/d05-x01-y01", # 50 bins from -2.5 to 2.5
                 # "/ATLAS_2010_S8918562/d12-x01-y01",  # 36 bins from 0.5 to 50
                # "/ATLAS_2010_S8918562/d19-x01-y01",  # 39 bins from 0.5 to 120.5
                # "/ATLAS_2010_S8918562/d25-x01-y01"   # 38 bins from 0.5 to 99.5
                # ]


        # def selectBinids(allbids, hnames):
            # good = []
            # for num, bid in enumerate(allbids):
                # # print("testing {}".format(bid))
                # for h in hnames:
                    # # print("\t for {}".format(h))
                    # if h in bid:
                        # # print("yes!")
                        # good.append(num)
                        # break
            # return good

        # selection = selectBinids(binids, S2_hnames)
        # binids = [binids[sel] for sel in selection]
        # RA = [RA[sel] for sel in selection]

        FMIN = [r.fmin(10) for r in RA]
        FMAX = [r.fmax(10) for r in RA]

        # Filter here to use only certain bins/histos 
        # TODO put the filtering in readApprox?
        Y, E = readExpData(sys.argv[2], [str(b) for b in  binids])

        # Also filter for not enveloped data
        good, bad = [],[]
        for num, bid in enumerate(binids):
            if FMIN[num]<= Y[num] and FMAX[num]>= Y[num]: good.append(num)
            else: bad.append(bid)

        RA     = [RA[g]     for g in good]
        binids = [binids[g] for g in good]
        E      = E[good]
        Y      = Y[good]

        E2 = [e**2 for e in E]
        SCLR = RA[0]._scaler
    else:
        Y=None
        E=None
        RA=None
        binids=None
        E2=None
        SCLR=None

    Y      = comm.bcast(Y     , root=0)
    E      = comm.bcast(E     , root=0)
    RA     = comm.bcast(RA    , root=0)
    binids = comm.bcast(binids, root=0)
    E2     = comm.bcast(E2    , root=0)
    SCLR   = comm.bcast(SCLR  , root=0)


    print("[{}] len bids: {}".format(rank, len(binids)))

    def chi2(x):
        return sum([ (Y[i] - RA[i](x))**2/E2[i] for i in range(len(binids))])

    def startPoint(ntrials):
        _PP = np.random.uniform(low=SCLR._Xmin,high=SCLR._Xmax,size=(ntrials,SCLR.dim))
        _CH = [chi2(p) for p in _PP]
        return _PP[_CH.index(min(_CH))]

    MULTISTART=10
    NSAMPLES = 1000

    if rank==0:
        t1=time.time()

        res = optimize.minimize(chi2, startPoint(MULTISTART), bounds=SCLR.box)
        t2=time.time()
        print("Minimum found at {} after {} seconds".format(res["x"], t2-t1))
    else:
        res=None


    # if rank==0:
        # from IPython import embed
        # embed()


    # Now do some more universes

    COV = mkCov(E)
    import scipy.stats as st

    # Here we draw samples using the Covariance matrix above
    # from IPython import embed
    # embed()
    try:
        mn = st.multivariate_normal(Y, COV)
    except Exception as e:
        print("Problem with COV: {} --- now trying to ignore singular values".format(e))
        mn = st.multivariate_normal(Y, COV, allow_singular=True)

    sampledyvals = mn.rvs(size=NSAMPLES).tolist()

    def chi2_smeared(x, V):
        for i in range(len(binids)):
            try:
                _ = V[i]
                __ = RA[i](x)
                ___ = E2[i]
            except:
                print(i, binids[i])
        return sum([ (V[i] - RA[i](x))**2/(2*E2[i]) for i in range(len(binids))])

    def startPointSmear(ntrials, v):
        _PP = np.random.uniform(low=SCLR._Xmin,high=SCLR._Xmax,size=(ntrials,SCLR.dim))
        _CH = [chi2_smeared(p, v) for p in _PP]
        return _PP[_CH.index(min(_CH))]

    if rank==0:
        allJobs=chunkIt(sampledyvals, size) # A list of lists of approximately the same length
    else:
        allJobs = []


    import sys
    rankJobs = comm.scatter(allJobs, root=0)
    print("[%i] sees %i items"%(rank, len(rankJobs)))
    sys.stdout.flush()

    res_smeared = []
    for num, v in enumerate(rankJobs):
        _r = optimize.minimize( lambda x:chi2_smeared(x, v), startPointSmear(MULTISTART, v), bounds=SCLR.box)
        res_smeared.append(_r)
        if (num+1)%10 == 0:
            print("[{}] Done with {}/{}".format(rank, num+1, len(rankJobs)))
            sys.stdout.flush()

    # Collective operation --- gather the long strings from each rank
    output = comm.gather(res_smeared, root=0)
    if rank==0:
        output.append([res]) # This is a list of lists
        P = [item["x"].tolist()   for sublist in output for item in sublist]
        F = [item["fun"] for sublist in output for item in sublist]

        with open("{}.minimization".format(sys.argv[1]), "w") as f:
            json.dump({ "x": P, "fun" : F, "scaler":SCLR.asDict}, f)
