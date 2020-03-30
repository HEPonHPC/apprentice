#!/usr/bin/python3

import apprentice as app
import numpy as np
from mpi4py import MPI


def objective(IO, x, comm, rankApp):

    a=IO._RA[0]
    xs = a._scaler.scale(x)
    rec_p = np.array(a.recurrence(xs, a._struct_p))

    v = np.array([IO._RA[i].predict(x, recurrence=rec_p) for i in rankApp])
    w = IO._W2[rankApp]
    y = IO._Y[rankApp]
    e = IO._E2[rankApp]
    d = y-v
    chi=np.sum(w*d*d*e)
    res=comm.allreduce(chi, op=MPI.SUM)
    return res

def startPoint(IO, ntrials, comm, rankApp, debug=False):
    _PP = np.random.uniform(low=IO._SCLR._Xmin,high=IO._SCLR._Xmax,size=(ntrials, IO._SCLR.dim))
    _CH = [objective(IO, p, comm, rankApp) for p in _PP]
    sp = _PP[_CH.index(min(_CH))]
    if debug:
        print("[{}] -- StartPoint: {}".format(comm.Get_rank(), sp ))
    return sp


def minimize(IO, comm, rankApp, ntrials, debug=False):
    from scipy import optimize
    x0 = startPoint(IO, ntrials, comm, rankApp)

    # for num, x0 in enumerate(SP):
    res = optimize.minimize(lambda x:objective(IO,x,comm,rankApp), x0, bounds=IO._bounds)
    return res


if __name__ == "__main__":
    import sys
    from scipy import optimize
    import time

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    from apprentice.tools import TuningObjective

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTPUT", default="tune.json", help="Output file name (default: %default)")
    op.add_option("-w", dest="WEIGHTS", default=None, help="Obervable file (default: %default)")
    op.add_option("-d", dest="DATA", default=None, help="Data file to compare to (default: %default)")
    op.add_option("-t", "--trials", dest="NTRIALS", type=int, default=100, help="Number of points to sample to find startpoint (default: %default)")
    op.add_option("-r", "--restart", dest="NRESTART", type=int, default=1, help="Number of restarts (default: %default)")
    op.add_option("--seed", dest="SEED", type=int, default=1234, help="Random number seed (default: %default)")
    op.add_option("--filter", dest="FILTER", default=False, action='store_true', help="Filter bins that do no envelope data (default: %default)")
    op.add_option("--filter-n", dest="FILTERN", default=10, type=int, help="Number of multistarts when determining fmin/max (default: %default)")
    opts, args = op.parse_args()

    if opts.WEIGHTS is None:
        raise Exception("No weight file spefified -- use  -w on CL")
    if opts.DATA is None:
        raise Exception("No data file spefified -- use  -d on CL")

    np.random.seed(opts.SEED)
    IO = TuningObjective(opts.WEIGHTS, opts.DATA, args[0], debug=opts.DEBUG, cache_recursions=True)
    rankApp = app.tools.chunkIt([x for x in range(len(IO._RA))], size) if rank==0 else []
    rankApp = comm.scatter(rankApp, root=0)

    # c=minimize(IO, comm, rankApp, opts.NTRIALS, opts.DEBUG)
    # print("[{}]  fun -- {}".format(rank, c["fun"]))
    # d=IO.minimize(opts.NTRIALS)
    # if rank==0: print("    [{}]  fun -- {}".format(rank, d["fun"]))

    # sys.stdout.flush()


    results = []
    t0=time.time()
    for num in range(opts.NRESTART):
        if rank==0: print("[{}] {}/{} at {}".format(rank, num+1, opts.NRESTART, time.time()-t0))
        sys.stdout.flush()
        c=minimize(IO, comm, rankApp, opts.NTRIALS, opts.DEBUG)
        results.append(c)
    t1=time.time()
    print("[{}] is done".format(comm.Get_rank()))
    print("[{}] MPI time: {} second".format(rank, t1-t0))
    sys.stdout.flush()
    if rank==0:
        mm=[x["fun"] for x in results]
        winner = mm.index(np.min(mm))
        dd = {}
        dd["fun"] = results[winner].fun
        dd["x"] = results[winner].x.tolist()
        dd["pnames"] = IO.pnames

        print("Best fit value: {}\nBest fit point:".format(dd["fun"]))
        for pn, x in zip(IO.pnames, dd["x"]):
            print("{}\t{}".format(pn, x))

        import json
        with open(opts.OUTPUT, "w") as f:
            json.dump(dd, f, indent=4)
        tmpi=t1-t0
        # sys.stdout.flush()
        # t0=time.time()
        # for _ in range(opts.NRESTART):
            # d=IO.minimize(opts.NTRIALS)
        # t1=time.time()
        # print("single time: {} second".format(t1-t0))
        # print("MPI speed up: {}".format((t1-t0)/tmpi))
    else:
        pass


    exit(0)



    # At the moment, the object selection for the chi2 minimisation is
    # simply everything that is in the approximation file
    if rank ==0:
        binids, RA = app.tools.readApprox(args[0])
        if opts.DEBUG: print("[{}] initially we have {} bins".format(rank, len(binids)))
        hnames = [b.split("#")[0] for b in binids]
        bnums  = [int(b.split("#")[1]) for b in binids]

        for hn, bnum in zip(hnames, bnums):
            pathmatch_matchers = [(m,wstr) for m,wstr in matchers.items() if m.match_path(hn)]
            posmatch_matchers  = [(m,wstr) for (m,wstr) in pathmatch_matchers if m.match_pos(bnum)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0 #< NB. using last match
            weights.append(w)

        if opts.FILTER:
            FMIN = [r.fmin(opts.FILTERN) for r in RA]
            FMAX = [r.fmax(opts.FILTERN) for r in RA]
        else:
            FMIN=[-1e101 for r in RA]
            FMAX=[ 1e101 for r in RA]

        # Filter here to use only certain bins/histos 
        # TODO put the filtering in readApprox?
        dd = app.tools.readExpData(opts.DATA, [str(b) for b in  binids])
        Y  = np.array([dd[b][0] for b in binids])
        E  = np.array([dd[b][1] for b in binids])
        # Also filter for not enveloped data
        good = []
        for num, bid in enumerate(binids):
            if FMIN[num]<= Y[num] and FMAX[num]>= Y[num] and weights[num]>0 and E[num]>0: good.append(num)

        RA     = [RA[g]     for g in good]
        binids = [binids[g] for g in good]
        E      = E[good]
        Y      = Y[good]
        W2     = [w*w for w in np.array(weights)[good]]

        E2 = [1./e**2 for e in E]
        SCLR = RA[0]._scaler
    else:
        Y=None
        E=None
        RA=None
        binids=None
        E2=None
        SCLR=None
        W2=None

    Y      = comm.bcast(Y     , root=0)
    E      = comm.bcast(E     , root=0)
    RA     = comm.bcast(RA    , root=0)
    binids = comm.bcast(binids, root=0)
    E2     = comm.bcast(E2    , root=0)
    SCLR   = comm.bcast(SCLR  , root=0)
    W2     = comm.bcast(W2    , root=0)

    if opts.DEBUG: print("[{}] After filtering: len(binids) = {}".format(rank, len(binids)))

    # from numba import jit, njit
    # @jit(fastmath=True, parallel=True)
    def fast_chi(lW2, lY, lRA, lE2, nb):
        s=0
        for i in range(nb):
            s += lW2[i]*(lY[i] - lRA[i])*(lY[i] - lRA[i])*lE2[i]
        return s


    def chi2(x):
        return fast_chi(W2, Y, [RA[i](x) for i in range(len(binids))], E2 , len(binids))
        # return sum([ W2[i]*(Y[i] - RA[i](x))**2*E2[i] for i in range(len(binids))])

    def startPoint(ntrials):
        _PP = np.random.uniform(low=SCLR._Xmin,high=SCLR._Xmax,size=(ntrials,SCLR.dim))
        _CH = [chi2(p) for p in _PP]
        if opts.DEBUG: print("StartPoint: {}".format(_PP[_CH.index(min(_CH))]))
        return _PP[_CH.index(min(_CH))]


    if rank==0:
        t1=time.time()

        res = optimize.minimize(chi2, startPoint(opts.NSTART), bounds=SCLR.box)
        t2=time.time()
        print("Minimum found at\n\t{}\nafter {} seconds".format("\n\t".join([ "{} {}".format(a,b) for a, b in zip(SCLR.pnames, res["x"])]), t2-t1))
    else:
        res=None

    # Write out results and make plot
    hnames = sorted(list(set([b.split("#")[0] for b in binids])))


    Y_tune = {}

    binids, RA = app.tools.readApprox(args[0])
    for h in hnames:
        _bins = [b for b in binids if h in b]
        _bins.sort(key = lambda x: int(x.split("#")[1]))
        Y_min = [RA[binids.index(b)].predict(res["x"]) for b in _bins]
        Y_tune.update([(a,b) for a, b in zip(_bins, Y_min)])

    # TODO Add more stuff like input file names, startpoint etc.
    d_out = {"Y":Y_tune, "X": res["x"].tolist(), "pnames":SCLR.pnames, "COV":res['hess_inv'].todense().tolist()}


    import json
    with open(opts.OUTPUT, "w") as f:
        json.dump(d_out, f, indent=4)



    import sys
    sys.exit(0)


    NSAMPLES = 100

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
            _ = V[i]
            __ = RA[i](x)
            ___ = E2[i]
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
