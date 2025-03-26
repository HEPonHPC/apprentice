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

def lineScan(TO, x0, dim, npoints=100, bounds=None):
    if bounds is None:
        xmin, xmax = TO.bounds_[TO.free_indices_][dim]
    else:
        xmin, xmax = bounds

    xcoords = list(np.linspace(xmin, xmax, npoints))
    xcoords.append(x0[dim])
    xcoords.sort()

    X = np.tile(x0, (len(xcoords),1))
    for num, x in enumerate(X):
        x[dim] = xcoords[num]
    return X

def mkPlotsMinimum(TO, x0, y0=None, prefix=""):
    import pylab
    pnames = np.array(TO.fnspace_.pnames)[TO.free_indices_]

    XX = [lineScan(TO,x0,i) for i in range(len(x0))]
    YY = []
    for i, X in enumerate(XX):
        Y   =[TO.objective(x) for x in X]
        YY.append(Y)
        Y   =[TO.objective(x, unbiased=True) for x in X]
        YY.append(Y)
    ymax=np.max(np.array(YY))
    ymin=np.min(np.array(YY))

    for i in range(len(x0)):
        pylab.clf()
        X=lineScan(TO,x0,i)
        Y   =[TO.objective(x) for x in X]
        Yunb=[TO.objective(x, unbiased=True) for x in X]
        pylab.axvline(x0[i], label="x0=%.5f"%x0[i], color="k")
        y0=TO.objective(x0, unbiased=True)
        # pylab.axhline(y0, label="unbiased y0=%.2f"%y0, color="k")
        pylab.plot(X[:,i], Y, label="objective")
        # pylab.plot(X[:,i], Yunb, linestyle="dashed", label="unbiased")
        #TODO just normalized ratio of bias vs unbiased
        pylab.ylabel("objective")
        pylab.xlabel(pnames[i])
        pylab.ylim((0.9*ymin, 1.1*ymax))
        # if abs(ymin-ymax)>1000:
            # pylab.yscale("log")
        pylab.legend()
        pylab.tight_layout()
        pylab.savefig(prefix+"valley_{}.pdf".format(i))

def mkPlotsCorrelation(TO, x0, prefix=""):
    H=TO.hessian(x0)
    COV = np.linalg.inv(H)
    std_ = np.sqrt(np.diag(COV))
    COR = COV / np.outer(std_, std_)

    nd = len(x0)
    mask =  np.tri(COR.shape[0], k=0)
    A = np.ma.array(COR, mask=mask)
    import pylab
    pylab.clf()
    bb = pylab.imshow(A, vmin=-1, vmax=1, cmap="RdBu")
    locs, labels = pylab.yticks()
    pylab.yticks([i for i in range(nd)], TO.fnspace_.pnames, rotation=00)
    locs, labels = pylab.xticks()
    pylab.xticks([i for i in range(nd)], TO.fnspace_.pnames, rotation=90)
    cbar = pylab.colorbar(bb, extend='both')

    pylab.tight_layout()
    pylab.savefig(prefix+"corr.pdf")

def printParams(TO, x):
    slen = max((max([len(p) for p in TO.fnspace_.pnames]), 6))
    from apprentice.appset import dot_aligned
    x_aligned = dot_aligned(x)
    plen = max((max([len(p) for p in x_aligned]), 6))

    b_dn = dot_aligned(TO.bounds_[:,0])
    b_up = dot_aligned(TO.bounds_[:,1])
    dnlen = max((max([len(p) for p in b_dn]), 5))
    uplen = max((max([len(p) for p in b_up]), 6))

    islowbound = x==TO.bounds_[:,0]
    isupbound  = x==TO.bounds_[:,1]
    isbound = islowbound + isupbound

    isbelow = x < TO.fnspace_.box[:,0]
    isabove = x > TO.fnspace_.box[:,1]
    isoutside = isbelow + isabove

    #isfixed = [i in self._fixIdx[0] for i in range(self.dim)]
    isfixed = [i in TO.fixed_indices_ for i in range(TO.fnspace_.dim_)]

    s= ""
    s+= ("#\n#{:<{slen}}\t{:<{plen}} #    COMMENT    [ {:<{dnlen}}  ...  {:<{uplen}} ]\n#\n".format(" PNAME", " PVALUE", " PLOW", " PHIGH", slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
    for pn, val, bdn, bup, isf, isb, iso in zip(TO.fnspace_.pnames, x_aligned, b_dn, b_up, isfixed, isbound, isoutside):

        if isb and isf:
            comment = "FIX & ONBOUND"
        elif isb and not isf:
            comment="ONBOUND"
        elif not isb and isf:
            comment="FIX"
        elif iso and not isf:
            comment = "OUTSIDE"
        elif iso and isf:
            comment = "FIX & OUTSIDE"
        else:
            comment = ""
        s+= ("{:<{slen}}\t{:<{plen}} # {:<13} [ {:<{dnlen}}  ...  {:<{uplen}} ]\n".format(pn, val, comment, bdn, bup, slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
    return s

def writeResult(TO, x, fname, meta=None):
    with open(fname, "w") as f:
        if meta is not None:
           f.write("{}".format(meta))
        f.write("{}".format(printParams(TO,x)))

if __name__ == "__main__":
    import optparse, os, sys, h5py

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTDIR", default="tune", help="Output directory (default: %default)")
    op.add_option("-e", "--errorapprox", dest="ERRAPP", default=None, help="Approximations of bin uncertainties (default: %default)")
    op.add_option("-s", dest="SEED", type=int, default=1234, help="Random seed (default: %default)")
    op.add_option("-r", "--restart", dest="RESTART", default=1, type=int, help="Minimiser restarts (default: %default)")
    op.add_option("--msp", dest="MSP", default=None, help="Manual startpoint, comma separated string (default: %default)")
    op.add_option("-a", "--algorithm", dest="ALGO", default="tnc", help="The minimisation algorithm tnc, ncg, lbfgsb, trust (default: %default)")
    op.add_option("-l", "--limits", dest="LIMITS", default=None, help="Parameter file with limits and fixed parameters (default: %default)")
    op.add_option("-f", dest="FORCE", default=False, action = 'store_true', help="Overwrite output directory (default: %default)")
    op.add_option("-p", "--plotvalley", dest="PLOTVALLEY", default=False, action = 'store_true', help="Parameter dependence near minimum (default: %default)")
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

    ps = PolySet.from_surrogates(APPR)
    eps = PolySet.from_surrogates(EAPP) if opts.ERRAPP else None

    DIM = ps.fnspace_.dim
    PNAMES = ps.fnspace_.pnames

    fixed = []
    bounds = ps.fnspace_.box.T
    if opts.LIMITS is not None:
        from apprentice.io import read_limitsandfixed
        # check that opts.LIMITS is a legitimate file
        lim, fix = read_limitsandfixed(opts.LIMITS)
        # fixed parameters not working
        fixed = [ [PNAMES.index(k), fix[k]] for k in fix.keys()]
        lims  = [ [PNAMES.index(k), list(lim[k])] for k in lim.keys()]
        for i, b in lims:
            bounds[0][i] = b[0]
            bounds[1][i] = b[1]


    GG = GeneratorTuning(DIM, ps.fnspace_, DATA, ERRS, ps, eps, WGT, BINS, BLOW, BUP, bounds=bounds, fixed=fixed)


    SC = ScipyMinimizer(GG,method=opts.ALGO)

    box = APPR[0].fnspace.box
    x0 = []
    if opts.MSP is not None:
        x0 = [float(x) for x in opts.MSP.split(",")]
    else:
        for i,b in enumerate(box):
            x = np.random.uniform(b[0],b[1])
            x0.append(x)

    x0 = np.array(x0)

    if opts.RESTART > 1 : x0 = None

    t0 = time.time()
    res = SC.minimize(x0, method = opts.ALGO, nrestart = opts.RESTART )
    t1 = time.time()
    print(res)


    x0 = res.x
#    GG.setWeights(dict(zip(binids[good],wt_flat)))
    chi2 = GG.objective(res.x, unbiased = True)
    ndf = len(WGT) - len(x0.flatten()) + 1

    meta  = "# Objective value at best fit point: %.2f (%.2f without weights)\n"%(res.fun, chi2)
    meta += "# Degrees of freedom: {}\n".format(ndf)
    meta += "# phi2/ndf: %.3f\n"%(chi2/ndf)
    meta += "# Minimisation took {} seconds\n".format(t1-t0)
    meta += "# Command line: {}\n".format(" ".join(sys.argv))
    meta += "# Best fit point:"

    print(meta)
    print(res.x)
    print(printParams(GG, res.x))

    DX = (bups - blows)*0.5
    X  = blows + DX
    Y  = [s.f_x(res.x) for s in AP]
    if EA==None:
        dY = np.zeros_like(Y)
    else:
        dY = [s.f_x(res.x) for s in EA]

    import yoda

    observables = np.unique( hnames )

    Y2D = list()
    for obs in observables:
        idx = np.where( np.array(hnames) == obs )
        try:
            P2D = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]
        except:
            P2D = [yoda.Point2D(x,y,dx,dy, source=b'') for x,y,dx,dy in zip(X[idx], np.array(Y)[idx], DX[idx], np.array(dY)[idx])]

        Y2D.append(yoda.Scatter2D(P2D, obs, obs))

    outcommon = "{}_{}_{}".format(opts.ALGO, opts.RESTART, opts.SEED)
    fileNameYoda = os.path.join(opts.OUTDIR, "predictions_{}.yoda".format(outcommon))
    yoda.write(Y2D, fileNameYoda)
    mkPlotsCorrelation(GG, res.x, opts.OUTDIR+"/{}_".format(outcommon))
    writeResult(GG, res.x, os.path.join(opts.OUTDIR, "minimum_{}.txt".format(outcommon)), meta=meta)
    if opts.PLOTVALLEY:
        plotout = os.path.join(opts.OUTDIR, "valleys_{}".format(outcommon))
        if not os.path.exists(plotout): os.makedirs(plotout)
        mkPlotsMinimum(GG, res.x, prefix=plotout+"/")

