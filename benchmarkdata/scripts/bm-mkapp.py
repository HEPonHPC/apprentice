#!/usr/bin/env python

import numpy as np
import apprentice as app

def mkAndStoreRapp(fin, fout, strategy, order, trainingsize):
    X, Y = app.readData(fin)
    if trainingsize <= Y.size:
        X=X[0:trainingsize]
        Y=Y[0:trainingsize]
    else:
        raise Exception("Requested trainigsize {} exceeds available data {}".format(trainingsize, Y.size))
    r=app.RationalApproximation(X,Y, order=order, strategy=strategy)
    r.save(fout)

def mkOutName(outdir, fin, strategy, order, trainingsize):
    import os
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise Exception("Specified output directory {} is an existing file.".format(outdir))
    if not os.path.exists(outdir): os.makedirs(outdir)

    fout = os.path.basename(fin).rsplit(".", 1)[0]
    fout += "_s{}_m{}_n{}_t{}.json".format(strategy, order[0], order[1], trainingsize)
    return os.path.join(outdir, fout)

if __name__=="__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-O", dest="OUTDIR", default=None, help="Output directory with automatic output file name generation (default: %default)")
    op.add_option("-o", dest="OUTFILE", default="app.json", help="Output file name (default: %default)")
    op.add_option("-s", dest="STRATEGY", default=2, type=int, help="Rational approximation strategy (default: %default)")
    op.add_option("-m", dest="M", default=1, type=int, help="Numerator polynomial order m (default: %default)")
    op.add_option("-n", dest="N", default=1, type=int, help="Denominator polynomial order m (default: %default)")
    op.add_option("-t", dest="TRAIN", default=-1, type=int, help="Training size (default: %default i.e. all available)")
    opts, args = op.parse_args()

    if opts.OUTDIR is None:
        mkAndStoreRapp(args[0], opts.OUTFILE, opts.STRATEGY, (opts.M, opts.N), opts.TRAIN)
    else:
        fout = mkOutName(opts.OUTDIR, args[0], opts.STRATEGY, (opts.M, opts.N), opts.TRAIN)
        mkAndStoreRapp(args[0], fout, opts.STRATEGY, (opts.M, opts.N), opts.TRAIN)
