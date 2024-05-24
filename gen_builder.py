#!/usr/bin/env python3

from apprentice.function import Function
from apprentice.polyset import PolySet
from apprentice.polynomialapproximation import PolynomialApproximation
from apprentice.rationalapproximation import RationalApproximation
from apprentice.generatortuning import GeneratorTuning
from apprentice.scipyminimizer import ScipyMinimizer
from apprentice.weights import read_pointmatchers

from apprentice.util import Util
import apprentice.io as IO
import apprentice.tools as TOOLS
import numpy as np
import time



if __name__ == "__main__":
    import optparse, os, sys, h5py
    #https://stackoverflow.com/questions/12200580/numpy-function-for-simultaneous-max-and-min
    import numba

    @numba.jit
    def minmax(x):
        maximum = x[0]
        minimum = x[0]
        for i in x[1:]:
            if i > maximum:
                maximum = i
            elif i < minimum:
                minimum = i
        return (minimum, maximum)

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTPUT", default="approx.json", help="Output filename (default: %default)")
    op.add_option("-s", "--seed", dest="SEED", type=int, default=1234, help="Random seed (default: %default)")
    op.add_option("-r", "--restart", dest="RESTART", default=1, type=int, help="Minimiser restarts (default: %default)")
    op.add_option("-a", "--algorithm", dest="ALGO", default="tnc", help="The minimisation algrithm tnc, ncg, lbfgsb, trust (default: %default)")
    op.add_option("-l", "--limits", dest="LIMITS", default=None, help="Parameter file with limits and fixed parameters (default: %default)")
    op.add_option("--msp",   dest="MSP", default=None, help="Manual startpoint, comma separated string (default: %default)")
    op.add_option("--order", dest="ORDER", default=None, help="Polynomial orders of numerator and denominator, comma separated (default: %default)")
    op.add_option("--mode",  dest="MODE", default="sip", help="Base algorithm  --- la |sip|lasip --- (default: %default)")
    op.add_option("--log",   dest="ISLOG", action='store_true', default=False, help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("--ftol",  dest="FTOL", type=float, default=1e-9, help="ftol for SLSQP (default: %default)")
    op.add_option("--errs",  dest="ERRS", action='store_true', default=False, help="Build approximations for errors, (default is for values)")

    opts,args = op.parse_args() 

    if opts.ALGO not in ["tnc", "ncg", "lbfgsb" ,"trust"]:
        raise Exception("Minimisation algorithm {} not implemented, should be tnc, ncg, lbfgsb or trust, exiting".format(opts.ALGO))


    rank=0
    size=1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception as e:
        print("Exception when trying to import mpi4py:", e)
        comm = None
        pass

    if len(args) == 0:
        print("No input specified, exiting")
        sys.exit(1)

    if not os.path.exists(args[0]):
        print("Input '{}' not found, exiting.".format(args[0]))
        sys.exit(1)

    # Prevent overwriting of input data
    assert(args[0]!=opts.OUTPUT)

    WFILE = args[0]
    MC    = args[1]

    if not os.path.isfile(args[0]):
        print("Input '{}' is not a file, exiting.".format(args[0]))
        sys.exit(1)

    if not os.path.isfile(args[1]):
        print("Input '{}' is not a file, exiting.".format(args[1]))
        sys.exit(1)

    DIN, binids, pnames, rankIdx, blows, bups = IO.readInputDataH5(MC, WFILE)

    hnames  = [    b.split("#")[0]  for b in binids]
    bnums   = [int(b.split("#")[1]) for b in binids]

    matchers = read_pointmatchers(WFILE)
    weights = []
    for hn, bnum, blow, bup in zip(hnames, bnums, blows, bups):
        pathmatch_matchers = [(m, wstr) for  m, wstr  in matchers.items()    if m.match_path(hn)]
        posmatch_matchers  = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum, blow, bup)]
        w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
        weights.append(w)
    weights = np.array(weights)
    nonzero = np.where( weights > 0 )

    import time
    t4 = time.time()
    import datetime

    binedges = {}
    pt_envelope = {}
    dapps = {}
    #amCache = {}
    APPR = []
    EAPP = []
    M = 3
    DIM = len(pnames)
    M, N = [int(x) for x in opts.ORDER.split(",")]
    nonzeroset = set(nonzero[0])

    #for num, (X, Y, E) in  enumerate(DIN):
    #    amCache[len(Y)] = False

    for num, (X, Y, E) in  enumerate(DIN):
        test = num in nonzeroset
        if not test :
            continue
        thisBinId = binids[num]
        xmin = blows[num]
        xmax = bups[num]
        if rank==0 or rank==size-1:
            #if ((num+1)%opts.MSGEVERY ==0):
            if ((num+1)% 10 ==0):
                now = time.time()
                tel = now - t4
                ttg = tel*(len(DIN)-num)/(num+1)
                eta = now + ttg
                eta = datetime.datetime.fromtimestamp(now + ttg)
                sys.stdout.write("{}[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(80*" " if rank>0 else "", rank, num+1, len(DIN), tel, ttg, eta.strftime('%Y-%m-%d %H:%M:%S')) ,)
                sys.stdout.flush()
        if len(X) == 0:
            print("No data to calculate approximation for {} --- skipping\n".format(binids[rankIdx[num]]))
            import sys
            sys.stdout.flush()
            continue
        if len(X) < TOOLS.numCoeffsRapp(len(X[0]),order=(M,N)):
            print("Not enough data ({} vs {}) to calculate approximation for {} --- skipping\n".format(len(X), app.tools.numCoeffsRapp(len(X[0]), order=(M,N)), binids[rankIdx[num]]))
            import sys
            sys.stdout.flush()
            continue

        if N > 0:
           print("include RA model")  
           sys.exit(1)
        else:
            if opts.ERRS:
                P = PolynomialApproximation.from_interpolation_points(
                    X, E, m = M, strategy = 1, pnames = pnames)
            else:
                P = PolynomialApproximation.from_interpolation_points(
                    X, Y, m = M, pnames = pnames, strategy = 1)
            APPR.append(P)
        dapps[thisBinId] = P.as_dict
        binedges[thisBinId] = (xmin,xmax)
        if not opts.ERRS: pt_envelope[thisBinId] = minmax(Y)

    if size > 1:
        DAPPS = comm.gather(dapps, root=0)
        DEDGE = comm.gather(binedges, root=0)
        if not opts.ERRS: DENVL = comm.gather(pt_envelope, root=0)
    else:
        DAPPS = [dapps]
        DEDGE = [binedges]
        if not opts.ERRS: DENVL = [pt_envelope]

    if rank == 0:
        from collections import OrderedDict
        JD = OrderedDict()

        a, e, r = {}, {}, {}
        for apps in DAPPS:
            a.update(apps)
        for edges in DEDGE:
            e.update(edges)
        if not opts.ERRS:
            for envls in DENVL:
                r.update(envls)

        xmin, xmax = [], []
        vmin, vmax = [], []
        for k in a.keys():
            JD[k] = a[k]
            xmin.append(e[k][0])
            xmax.append(e[k][1])
            if not opts.ERRS:
                vmin.append(r[k][0])
                vmax.append(r[k][1])
        JD["__xmin"] = xmin
        JD["__xmax"] = xmax
        if not opts.ERRS:
            JD["__vmin"] = vmin
            JD["__vmax"] = vmax

        import json
        with open(opts.OUTPUT, "w") as f: json.dump(JD,f, indent=4)

        sys.stdout.flush()
        print("Done")

    exit(0)
