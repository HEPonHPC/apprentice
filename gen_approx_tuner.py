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

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTPUT", default="approx.json", help="Output filename (default: %default)")
    op.add_option("-e", "--errorapprox", dest="ERRAPP", default=None, help="Approximations of bin uncertainties (default: %default)")
    op.add_option("-s", dest="SEED", type=int, default=1234, help="Random seed (default: %default)")
    op.add_option("-r", "--restart", dest="RESTART", default=1, type=int, help="Minimiser restarts (default: %default)")
    op.add_option("--msp", dest="MSP", default=None, help="Manual startpoint, comma separated string (default: %default)")
    op.add_option("-a", "--algorithm", dest="ALGO", default="tnc", help="The minimisation algrithm tnc, ncg, lbfgsb, trust (default: %default)")
    op.add_option("-l", "--limits", dest="LIMITS", default=None, help="Parameter file with limits and fixed parameters (default: %default)")
    op.add_option("--order", dest="ORDER", default=None, help="Polynomial orders of numerator and denominator, comma separated (default: %default)")
    op.add_option("--mode", dest="MODE", default="sip", help="Base algorithm  --- la |sip|lasip --- (default: %default)")
    op.add_option("--log", dest="ISLOG", action='store_true', default=False, help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("--ftol", dest="FTOL", type=float, default=1e-9, help="ftol for SLSQP (default: %default)")
    opts,args = op.parse_args() 

    if opts.ALGO not in ["tnc", "ncg", "lbfgsb" ,"trust"]:
        raise Exception("Minimisation algorithm {} not implemented, should be tnc, ncg, lbfgsb or trust, exiting".format(opts.ALGO))
    WFILE  = args[0]
    DATA   = args[1]
    APPROX = args[2]

    np.random.seed(opts.SEED)

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

    import json

    with open(APPROX) as f: rd = json.load(f)
    blows = rd["__xmin"]
    rd.pop('__xmin',None)
    bups  = rd["__xmax"]
    rd.pop('__xmax',None)
    if '__vmin' in rd.keys():
        vlows = rd["__vmin"]
        rd.pop('__vmin',None)
        vups  = rd["__vmax"]
        rd.pop('__vmax',None)

    if opts.ERRAPP != None:
        with open(opts.ERRAPP) as f: rde = json.load(f)
        EAPP = []
    else:
        EAPP = None


    binids = TOOLS.sorted_nicely( rd.keys() )

    blows = np.array(blows)
    bups = np.array(bups)

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
    binids = np.array(binids)

    nonzero = np.nonzero( weights )

    dexp = IO.readExpData(DATA,binids[nonzero])

    """
    Is dexp guaranteed to be ordered in the right way?   This was the issue before.
    """

    DATA = np.array([dexp[b][0] for b in binids[nonzero]], dtype=np.float64)
    ERRS = np.array([dexp[b][1] for b in binids[nonzero]], dtype=np.float64)

    # delay until later?
    good = np.nonzero(ERRS)

    APPR = []
    for b in binids:
        P_output = rd[b]
        P_from_data_structure = PolynomialApproximation.from_data_structure(P_output)
        APPR.append(P_from_data_structure)
        if opts.ERRAPP != None:
            PE_from_data_structure = PolynomialApproximation.from_data_structure(rde[b])
            EAPP.append(PE_from_data_structure)


    # Do envelope filtering
    ev = (DATA > vlows) & (DATA < vups) & (ERRS > 0)

    good = np.nonzero( ERRS * ev)

    # Hypothesis filtering

    # Any bins left?

    """
    WGT  = weights[nonzero]
    BLOW = blows[nonzero]
    BUP  = bups[nonzero]
    BINS = binids[nonzero]
    """
    AP   = APPR
    EA   = EAPP
    WGT  = weights[good]
    BLOW = blows[good]
    BUP  = bups[good]
    BINS = binids[good]
    DATA = DATA[good]
    ERRS = ERRS[good]
    APPR = [APPR[g] for g in good[0].flatten()]
    if opts.ERRAPP: EAPP = [EAPP[g] for g in good[0].flatten()]

    DIM = APPR[0].fnspace.dim_

    GG = GeneratorTuning(DIM, APPR[0].fnspace, DATA, ERRS, APPR, EAPP, WGT, BINS, BLOW, BUP)

    SC = ScipyMinimizer(GG,method=opts.ALGO)

    box = APPR[0].fnspace.box
    x0 = []
    for b in box:
        x = np.random.uniform(b[0],b[1])
        x0.append(x)
    x0 = np.array(x0)

    t0 = time.time()
    res = SC.minimize(x0)
    t1 = time.time()
    print(res)
    wt_flat = np.ones_like(WGT)
    GG.setWeights(dict(zip(binids[good],wt_flat)))
    chi2 = GG.objective(res.x)
    ndf = len(WGT) - len(x0.flatten()) + 1

    meta  = "# Objective value at best fit point: %.2f (%.2f without weights)\n"%(res.fun, chi2)
    meta += "# Degrees of freedom: {}\n".format(ndf)
    meta += "# phi2/ndf: %.3f\n"%(chi2/ndf)
    meta += "# Minimisation took {} seconds\n".format(t1-t0)
    meta += "# Command line: {}\n".format(" ".join(sys.argv))
    meta += "# Best fit point:"

    print(meta)
    print(res.x)

    DX = (bups - blows)*0.5
    X  = blows + DX
    Y  = [s.f_x(res.x) for s in AP]
    dY = [s.f_x(res.x) for s in EA]

    Y2D = []
    import yoda

    observables = np.unique( hnames )

    for obs in observables:
        idx = np.where( np.array(hnames) == obs )
        try:
            P2D = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]
        except:
            P2D = [yoda.Point2D(x,y,dx,dy, source=b'') for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]

        Y2D.append(yoda.Scatter2D(P2D, obs, obs))
    yoda.write(Y2D,'test.yoda')