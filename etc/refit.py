import apprentice as app
import numpy as np


####
# * Store original boxes?
# * allow user defined target domain
# * check dependency on a,b
# * extend to rationals
# * what to do about vmin, vmax ...
# * mpi?

if __name__ == "__main__":
    import sys

    assert(sys.argv[1]!=sys.argv[2])
    binids, P = app.io.readApprox(sys.argv[1])

    SC = [p._scaler for p in P]

    DIM = SC[0].dim

    xmin, xmax = [], []
    for d in range(DIM):
        xmin.append(max([s.box[d][0] for s in SC]))
        xmax.append(min([s.box[d][1] for s in SC]))

    # New scaler
    sc = app.Scaler([xmin, xmax])
    sc._pnames = SC[0].pnames

    NC = app.tools.numCoeffsPoly(P[0].dim, P[0].m)
    X  = P[0]._scaler.drawSamples(NC)
    A  = np.prod(np.power(sc.scale(X), P[0]._struct_p[:, np.newaxis]), axis=2).T

    Z = []
    import time
    t0=time.time()
    for num, p in enumerate(P):
        # cnew = app.tools.refitPoly(p, sc)
        cnew = app.tools.refitPolyAX(p, A, X)
        Z.append(cnew)
        if (num+1)%50==0:
            print("{}/{} after {} seconds".format(num+1, len(P), time.time()-t0))
    print("That took {} seconds".format(time.time() - t0))

    for z, p in zip(Z,P):
        p._scaler=sc
        p._pcoeff=z

    import json
    with open(sys.argv[1]) as f:
        rd = json.load(f)

    for b, p in zip(binids, P):
        rd[b] = p.asDict

    with open(sys.argv[2], "w") as f:
        json.dump(rd, f, indent=4)


    # TEST
    binidsold, Pold = app.io.readApprox(sys.argv[1])
    binidsnew, Pnew = app.io.readApprox(sys.argv[2])

    OLD = {b:p for b, p in zip(binidsold, Pold)}
    NEW = {b:p for b, p in zip(binidsnew, Pnew)}

    for b, p in OLD.items():
        x = p._scaler.center
        print("{} vs {}, diff: {}".format(p(x), NEW[b](x), p(x) - NEW[b](x)))
