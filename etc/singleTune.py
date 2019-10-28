#!/usr/bin/python3

import apprentice as app
import numpy as np
from mpi4py import MPI

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-c", "--cache", dest="CACHE", action="store_true", default=False, help="Cache stuff (experimental)")
    op.add_option("-o", dest="OUTPUT", default="singletune.json", help="Output file name (default: %default)")
    op.add_option("-w", dest="WEIGHTS", default=None, help="Obervable file (default: %default)")
    op.add_option("-d", dest="DATA", default=None, help="Data file to compare to (default: %default)")
    op.add_option("-t", "--trials", dest="NTRIALS", type=int, default=10, help="Number of points to sample to find startpoint (default: %default)")
    op.add_option("-r", "--restart", dest="NRESTART", type=int, default=10, help="Number of restarts (default: %default)")
    op.add_option("--seed", dest="SEED", type=int, default=1234, help="Random number seed (default: %default)")
    op.add_option("--filter", dest="FILTER", default=False, action='store_true', help="Filter bins that do no envelope data (default: %default)")
    op.add_option("--filter-n", dest="FILTERN", default=10, type=int, help="Number of multistarts when determining fmin/max (default: %default)")
    opts, args = op.parse_args()

    if opts.WEIGHTS is None:
        raise Exception("No weight file spefified -- use  -w on CL")
    if opts.DATA is None:
        raise Exception("No data file spefified -- use  -d on CL")



    np.random.seed(opts.SEED)
    import time
    t0=time.time()
    IO  = app.tools.TuningObjective(opts.WEIGHTS, opts.DATA, args[0], debug=opts.DEBUG, cache_recursions=opts.CACHE)
    t1=time.time()
    print("[{}] Build took {} seconds".format(rank, t1-t0))
    sys.stdout.flush()

    rankWork = app.tools.chunkIt(IO._hnames, size) if rank==0 else []
    rankWork = comm.scatter(rankWork, root=0)


    d={}
    t0=time.time()
    for num, h in enumerate(rankWork):
        _ = IO.minimize(opts.NTRIALS, nrestart=opts.NRESTART, sel=IO.obsBins(h))
        d[h] = [_.fun] + _.x.tolist()
        if (num+1)%10 ==0:
            print("[{}] {}/{} after {} seconds".format(rank, num+1, len(rankWork), time.time() - t0))
            sys.stdout.flush()

    comm.barrier()
    L=comm.gather(d, root=0)
    if rank==0:
        results = {k: v for d in L for k, v in d.items()}
        with open(opts.OUTPUT, "w") as f:
            import json
            json.dump(results, f, indent=4)
