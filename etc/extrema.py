#!/usr/bin/python3

import apprentice as app
import numpy as np
from mpi4py import MPI


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
    op.add_option("-o", dest="OUTPUT", default="approxwextremata.json", help="Output file (default: %default)")
    op.add_option("-t", "--trials", dest="NTRIALS", type=int, default=100, help="Number of points to sample to find startpoint (default: %default)")
    op.add_option("-r", "--restart", dest="NRESTART", type=int, default=1, help="Number of restarts (default: %default)")
    op.add_option("--seed", dest="SEED", type=int, default=1234, help="Random number seed (default: %default)")
    op.add_option("-i", "--inplace", dest="INPLACE", default=False, action='store_true', help="Overwrite input file (default: %default)")
    # op.add_option("-g", "--grad", dest="USEGRAD", default=False, action='store_true', help="Use gradients in minimisation (default: %default)")
    opts, args = op.parse_args()

    np.random.seed(opts.SEED)
    binids, RA = app.tools.readApprox(args[0], set_structures = True)

    rankWork = app.tools.chunkIt([i for i in range(len(binids))], size) if rank==0 else []
    rankWork = comm.scatter(rankWork, root=0)
    import time
    import sys
    import datetime
    t0=time.time()


    FMIN, FMAX = [], []
    for i in rankWork:
        FMIN.append((binids[i], RA[i].fmin(opts.NTRIALS, opts.NRESTART, use_grad=True)))
        FMAX.append((binids[i], RA[i].fmax(opts.NTRIALS, opts.NRESTART, use_grad=True)))

        if rank==0:
            now = time.time()
            tel = now - t0
            ttg = tel*(len(rankWork)-i)/(i+1)
            eta = now + ttg
            eta = datetime.datetime.fromtimestamp(now + ttg)
            sys.stdout.write("[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(rank, i+1, len(rankWork), tel, ttg, eta.strftime('%Y-%m-%d %H:%M:%S')))
            sys.stdout.flush()
        sys.stdout.flush()

    if rank==0:
        print()

    # FMIN = [(binids[i], RA[i].fmin(opts.NTRIALS, use_grad=opts.USEGRAD)) for i in rankWork]
    # FMAX = [(binids[i], RA[i].fmax(opts.NTRIALS, use_grad=opts.USEGRAD)) for i in rankWork]
    t1=time.time()

    _FMIN = comm.gather(FMIN, root=0)
    _FMAX = comm.gather(FMAX, root=0)
    if rank==0:
        _FMIN = [ x for sublist in _FMIN for x in sublist]
        _FMAX = [ x for sublist in _FMAX for x in sublist]
        for k, v in _FMIN:
            i = binids.index(k)
            RA[i]._vmin=v
        for k, v in _FMAX:
            i = binids.index(k)
            RA[i]._vmax=v

        import json
        with open(args[0]) as f:
            rd = json.load(f)
            xmin = rd["__xmin"]
            xmax = rd["__xmax"]

        JD = {b : r.asDict for b,r in zip(binids, RA)}
        JD["__xmin"] = xmin
        JD["__xmax"] = xmax

        if not opts.INPLACE:
            with open(opts.OUTPUT, "w") as f: json.dump(JD, f, indent=4)
            print("Done --- {} approximations written to {}".format(len(JD), opts.OUTPUT))
        else:
            with open(args[0], "w") as f: json.dump(JD, f, indent=4)
            print("Done --- {} approximations written in place to {}".format(len(JD), args[0]))
