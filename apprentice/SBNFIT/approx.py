import apprentice
import argparse
import h5py
import numpy as np
import sys,os
import random
from timeit import default_timer as timer
"""
-i ../../../log/SBNFIT/comparespectrum_mpi_deg2.h5 -o ../../../log/SBNFIT/approx -m 2 -n 0 -t Cp
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Approx for SBNFIT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    parser.add_argument("-o", "--outtdir", dest="OUTDIR", type=str, default=None,
                        help="Output Dir")
    parser.add_argument("-s", "--solver", dest="SOLVER", type=str, default="ipopt",
                        help="Output Dir")
    parser.add_argument("-m", "--numerorder", dest="M", type=int, default=2,
                        help="Numerator order")
    parser.add_argument("-n", "--denomorder", dest="N", type=int, default=1,
                        help="Denominator order")
    parser.add_argument("-t", "--trainsize", dest="TRAINSIZE", type=str, default="1x",
                        help="Training size 1x, 2x, or Cp")


    args = parser.parse_args()

    import time

    t1 = time.time()

    f = h5py.File(args.INFILE, "r")
    data = np.array(f.get('colspec'))
    f.close()
    # print(np.shape(data))
    nbin = np.shape(data)[1]
    npoints = np.shape(data)[0]
    npointsperdim = int(np.sqrt(npoints))
    # print(npoints,nbin,npointsperdim)

    Xall = []
    for i in range(npointsperdim):
        for j in range(npointsperdim):
          Xall.append([i,j])
    # print(len(Xall))

    np.random.seed(83772)
    size = apprentice.tools.numCoeffsPoly(2,args.M) + apprentice.tools.numCoeffsPoly(2,args.N)
    print(size)
    X = []
    if args.TRAINSIZE == '1x':
        tsize = size
        tindex = random.sample(range(npoints), tsize)
        for i in tindex:
            X.append(Xall[i])
    elif args.TRAINSIZE == '2x':
        tsize = 2*size
        tindex = random.sample(range(npoints), tsize)
        for i in tindex:
            X.append(Xall[i])
    elif args.TRAINSIZE == 'Cp':
        tsize = npoints
        tindex = range(npoints)
        X = Xall
    else: raise Exception('huh?')


    binids = ["Bin{}".format(i) for i in range(nbin)]
    pnames = ['p1','p2']

    apps = []
    if args.N > 0:
        from shutil import which
        if which(args.SOLVER) is None:
            print("AMPL solver {} not found in PATH, exiting".format(args.SOLVER))
            sys.exit(1)
    import time

    t4 = time.time()
    import datetime

    binedges = {}
    dapps = {}
    amCache = {}


    for bno in range(nbin):
        amCache[tsize] = False
        thisBinId = binids[bno]
        Yall = data[:,bno]

        Y = []
        for i in tindex:
            Y.append(Yall[i])
        starttime = timer()
        temp, hasPole = apprentice.tools.calcApprox(X, Y, (args.M, args.N), pnames=pnames,
                                             mode="sip", debug=False,
                                             testforPoles=10,
                                             ftol=1e-9, itslsqp=200,
                                             solver=args.SOLVER,
                                             abstractmodel=False, tmpdir="/tmp")
        vmin = apprentice.tools.extreme(temp,nsamples=10,nrestart=100,use_grad=True,mode="min")
        vmax = apprentice.tools.extreme(temp,nsamples=10,nrestart=100,use_grad=True,mode="max")
        try:
            if amCache[len(X)] is None:
                amCache[len(X)] = temp._abstractmodel
        except Exception as e:
            print("AM no worky: {}".format(e))

        if temp is None:
            print("Unable to calculate value approximation for {} --- skipping\n".format(thisBinId))
            import sys
            sys.stdout.flush()
            continue
        else:
            if hasPole:
                print("Warning: pole detected in {}\n".format(thisBinId))
                import sys
                sys.stdout.flush()
        temp._vmin = float(vmin)
        temp._vmax = float(vmax)
        dapps[thisBinId] = temp.asDict
        dapps[thisBinId]['log'] = {'time':timer()-starttime}
        # print(dapps[thisBinId])
        print("Done with {} bins     \r".format(bno))

    # DAPPS = comm.gather(dapps, root=0)
    # t5 = time.time()
    # if rank==0:
    #     print()
    #     print("Approximation calculation took {} seconds".format(t5-t4))
    #     sys.stdout.flush()

    # Store in JSON
    from collections import OrderedDict

    # JD = OrderedDict()
    # a = {}
    # for apps in dapps:
    #     a.update(apps)
    # for k in a.keys():
    #     JD[k] = a[k]
    import json

    os.makedirs(args.OUTDIR,exist_ok=True)
    appfile = os.path.join(args.OUTDIR,"approximation_m{}_n{}_t{}.json".format(args.M,args.N,args.TRAINSIZE))
    trainfile = os.path.join(args.OUTDIR,
                           "training_m{}_n{}_t{}.json".format(args.M, args.N, args.TRAINSIZE))
    traindata = {}
    traindata['trainindex'] = [i for i in tindex]
    traindata['Xtrain'] = X
    with open(appfile, "w") as f:
        json.dump(dapps, f, indent=4)
    with open(trainfile, "w") as f:
        json.dump(traindata, f, indent=4)





