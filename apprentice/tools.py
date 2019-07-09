# Documentation?
def fast_chi(lW2, lY, lRA, lE2, nb):
    s=0
    for i in range(nb):
        s += lW2[i]*(lY[i] - lRA[i])*(lY[i] - lRA[i])*lE2[i]
    return s

def numNonZeroCoeff(app, threshold=1e-6):
    """
    Determine the number of non-zero coefficients for an approximation app.
    """
    n=0
    for p in app._pcoeff:
        if abs(p)>threshold: n+=1

    if hasattr(app, '_qcoeff'):
        for q in app._qcoeff:
            if abs(q)>threshold: n+=1

    return n


def numNLPoly(dim, order):
    if order <2: return 0
    else:
        return numCoeffsPoly(dim, order) - numCoeffsPoly(dim, 1)

def numNL(dim, order):
    """
    Number of non-linearities.
    """
    m, n = order
    if n ==0 : return numNLPoly(dim, m)
    if m <2 and n <2: return 0
    elif m<2 and n>=2:
        return numCoeffsPoly(dim, n) - numCoeffsPoly(dim, 1)
    elif n<2 and m>=2:
        return numCoeffsPoly(dim, m) - numCoeffsPoly(dim, 1)
    else:
        return numCoeffsRapp(dim, order) -  numCoeffsRapp(dim, (1,1))

def numCoeffsPoly(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    ntok = 1
    r = min(order, dim)
    for i in range(r):
      ntok = ntok*(dim+order-i)/(i+1)
    return int(ntok)

def numCoeffsRapp(dim, order):
    """
    Number of coefficients a dim-dimensional rational approximation of order (m,n) has.
    """
    return numCoeffsPoly(dim, order[0]) + numCoeffsPoly(dim, order[1])

def maxOrder(N, dim):
    """
    Utility function to find highest order polynomial
    for dataset with N dim-dimensional points.
    Intended to be used when constructing max ONB
    """
    from scipy.special import comb
    omax = 0
    while comb(dim + omax+1, omax+1) + 1 <= N: # The '+1' stands for a order 0 polynomial's dof
        omax+=1
    return omax


def readH5(fname, idx=[0], xfield="params", yfield="values"):
    """
    Read X,Y values etc from HDF5 file.
    By default, only the first object is read.
    The X and Y-value dataset names depend on the file of course, so we allow
    specifying what to use. yfield can be values|errors with the test files.
    Returns a list of tuples of arrays : [ (X1, Y1), (X2, Y2), ...]
    The X-arrays are n-dimensional, the Y-arrays are always 1D
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f:
        indexsize = f.get("index").size

    # A bit of logic here --- if idx is passed an empty list, ALL data is read from file.
    # Otherwise we need to check that we are not out of bounds.

    # pnames = [p for p in f.get(xfield).attrs["names"]]
    if len(idx)>0:
        assert(max(idx) <= indexsize)
    else:
        idx=[i for i in range(indexsize)]

    ret = []
    f = h5py.File(fname, "r")

    # Read parameters
    _X=np.array(f.get(xfield))

    # Read y-values
    for i in idx:
        _Y=np.atleast_1d(f.get(yfield)[i])
        USE = np.where( (~np.isinf(_Y))  & (~np.isnan(_Y)) )
        ret.append([ _X[USE], _Y[USE] ])

    f.close()

    return ret

# TODO rewrite such that yfield is a list of datasetnames, e.g. yfield=["values", "errors"]

def readH52(fname, idx=[0], xfield="params", yfield1="values", yfield2="errors"):
    """
    Read X,Y, erros values etc from HDF5 file.
    By default, only the first object is read.
    The X and Y-value dataset names depend on the file of course, so we allow
    specifying what to use. yfield can be values|errors with the test files.
    Returns a list of tuples of arrays : [ (X1, Y1, E1), (X2, Y2, E2), ...]
    The X-arrays are n-dimensional, the Y-arrays are always 1D
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f:
        indexsize = f.get("index").size

    # A bit of logic here --- if idx is passed an empty list, ALL data is read from file.
    # Otherwise we need to check that we are not out of bounds.

    # pnames = [p for p in f.get(xfield).attrs["names"]]
    if len(idx)>0:
        assert(max(idx) <= indexsize)
    else:
        idx=[i for i in range(indexsize)]

    ret = []
    f = h5py.File(fname, "r")

    # Read parameters
    _X=np.array(f.get(xfield))

    # Read y-values
    for i in idx:
        _Y=np.atleast_1d(f.get(yfield1)[i])
        _E=np.atleast_1d(f.get(yfield2)[i])
        USE = np.where( (~np.isinf(_Y))  & (~np.isnan(_Y)) & (~np.isinf(_E))& (~np.isnan(_E)) )
        ret.append([ _X[USE], _Y[USE], _E[USE] ])

    f.close()

    return ret

def readPnamesH5(fname, xfield):
    """
    Get the parameter names from the hdf5 files params dataset attribute
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f:
        pnames = [p.astype(str) for p in f.get(xfield).attrs["names"]]

    return pnames

def readData(fname, delimiter=","):
    """
    Read CSV formatted data. The last column is interpreted as
    function values while all other columns are considered
    parameter points.
    """
    import os
    if not os.path.exists(fname): raise Exception("File {} not found".format(fname))
    import numpy as np
    D = np.loadtxt(fname, delimiter=delimiter)
    X=D[:,0:-1]
    Y=D[:,-1]
    USE = np.where( (~np.isinf(Y))  & (~np.isnan(Y)) )
    return X[USE], Y[USE]

def readApprentice(fname):
    """
    Read an apprentice JSON file. We abuse try except here to
    figure out whether it's a rational or polynomial approximation.
    """
    import apprentice
    import os
    if not os.path.exists(fname): raise Exception("File {} not found".format(fname))
    try:
        app = apprentice.RationalApproximation(fname=fname)
    except:
        app = apprentice.PolynomialApproximation(fname=fname)
    return app

def getPolyGradient(coeff, X, dim=2, n=2):
    from apprentice import monomial
    import numpy as np
    struct_q = monomial.monomialStructure(dim, n)
    grad = np.zeros(dim,dtype=np.float64)

    for coord in range(dim):
        """
        Partial derivative w.r.t. coord
        """
        der = [0.]
        if dim==1:
            for s in struct_q[1:]: # Start with the linear terms
                der.append(s*X[0]**(s-1))
        else:
            for s in struct_q[1:]: # Start with the linear terms
                if s[coord] == 0:
                    der.append(0.)
                    continue
                term = 1.0
                for i in range(len(s)):
                    # print(s[i])
                    if i==coord:
                        term *= s[i]
                        term *= X[i]**(s[i]-1)
                    else:
                        term *= X[i]**s[i]
                der.append(term)
        grad[coord] = np.dot(der, coeff)
    return grad

def possibleOrders(N, dim, mirror=False):
    """
    Utility function to find all possible polynomials
    orders for dataset with N points in N dimension
    """
    from scipy.special import comb
    omax = 0
    while comb(dim + omax+1, omax+1) + 1 <= N: # The '+1' stands for a order 0 polynomial's dof
        omax+=1

    combs = []
    for m in reversed(range(omax+1)):
        for n in reversed(range(m+1)):
            if comb(dim + m, m) + comb(dim+n,n) <= N:
                combs.append((m,n))

    if mirror:
        temp=[tuple(reversed(i)) for i in combs]
        for t in temp:
            if not t in combs:
                combs.append(t)
    return combs

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
# Todo add binwidth in data model
def readExpData(fname, binids):
    import json
    import numpy as np
    with open(fname) as f: dd = json.load(f)
    Y = np.array([dd[b][0] for b in binids])
    E = np.array([dd[b][1] for b in binids])
    return dict([(b, (y, e)) for b,y,e in zip(binids,Y,E)])

def readTuneResult(fname):
    import json
    with open(fname) as f:
        return json.load(f)

def readApprox(fname):
    import json, apprentice
    with open(fname) as f: rd = json.load(f)
    binids = sorted(rd.keys())
    APP = {}
    for b in binids:
        try:
            APP[b]=apprentice.RationalApproximation(initDict=rd[b])
        except:
            APP[b]=apprentice.PolynomialApproximation(initDict=rd[b])
    return binids, [APP[b] for b in binids]

def mkCov(yerrs):
    import numpy as np
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])


class TuningObjective(object):
    def __init__(self, f_weights, f_data, f_approx, restart_filter=None, debug=False):
        import apprentice
        import numpy as np
        matchers=apprentice.weights.read_pointmatchers(f_weights)
        binids, RA = apprentice.tools.readApprox(f_approx)
        if debug: print("Initially we have {} bins".format(len(binids)))
        hnames = [b.split("#")[0] for b in binids]
        bnums  = [int(b.split("#")[1]) for b in binids]

        self._debug = debug

        weights = []
        for hn, bnum in zip(hnames, bnums):
            pathmatch_matchers = [(m,wstr) for m,wstr in matchers.items() if m.match_path(hn)]
            posmatch_matchers  = [(m,wstr) for (m,wstr) in pathmatch_matchers if m.match_pos(bnum)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0 #< NB. using last match
            weights.append(w)

        # TODO This should be passed from the outside for performance
        if restart_filter is not None:
            FMIN = [r.fmin(restart_filter) for r in RA]
            FMAX = [r.fmax(restart_filter) for r in RA]
        else:
            FMIN=[-1e101 for r in RA]
            FMAX=[ 1e101 for r in RA]

        # TODO This needs to be a porperty
        # Filter here to use only certain bins/histos
        # TODO put the filtering in readApprox?
        dd = apprentice.tools.readExpData(f_data, [str(b) for b in  binids])
        Y  = np.array([dd[b][0] for b in binids])
        E  = np.array([dd[b][1] for b in binids])
        # Also filter for not enveloped data
        good = []
        for num, bid in enumerate(binids):
            if FMIN[num]<= Y[num] and FMAX[num]>= Y[num] and weights[num]>0: good.append(num)

        # TODO --- upon calling a def filter() --- these should be properties
        self._RA     = [RA[g]     for g in good]
        self._binids = [binids[g] for g in good]
        self._E      = E[good]
        self._Y      = Y[good]
        self._W2     = np.array([w*w for w in np.array(weights)[good]])

        self._E2 = np.array([1./e**2 for e in self._E])
        self._SCLR = RA[0]._scaler # Replace with min/max limits things
        self._hnames = sorted(list(set([b.split("#")[0] for b in self._binids])))
        hdict = dict([(h, []) for h in self._hnames])
        for b in self._binids:
            hname, bid = b.split("#")
            hdict[hname].append(bid)
        self._hdict = hdict

        if debug: print("After filtering: len(binids) = {}".format(len(self._binids)))

    def setWeights(self, wdict):
        """
        Convenience function to update the bins weights.
        """
        for num, b in enumerate(self._binids):
            for hn, w in wdict.items():
                if hn in b:
                    self._W2[num] = w*w

    def obsGoF(self, hname, x):
        """
        Convenience function to get the (unweighted) contribution to the gof for obs hname at point x
        """
        import numpy as np
        _bids = [num for num, b in enumerate(self._binids) if hname in b]
        return fast_chi(np.ones(len(_bids)), self._Y[_bids], [self._RA[i](x) for i in _bids], self._E2[_bids] , len(_bids))

    def meanCont(self, x):
        """
        Convenience function that return a list if the obsGof for all hnames at x
        """
        return [self.obsGoF(hn, x) for hn in self.hnames]

    @property
    def hnames(self): return self._hnames

    @property
    def pnames(self): return self._SCLR.pnames

    def objective(self, x):
        return fast_chi(self._W2, self._Y, [f(x) for f in self._RA], self._E2 , len(self._binids))

    def startPoint(self, ntrials):
        import numpy as np
        _PP = np.random.uniform(low=self._SCLR._Xmin,high=self._SCLR._Xmax,size=(ntrials, self._SCLR.dim))
        _CH = [self.objective(p) for p in _PP]
        if self._debug: print("StartPoint: {}".format(_PP[_CH.index(min(_CH))]))
        return _PP[_CH.index(min(_CH))]

    def minimize(self, nstart):
        from scipy import optimize
        res = optimize.minimize(self.objective, self.startPoint(nstart), bounds=self._SCLR.box)
        return res

class Outer(TuningObjective):

    def setWeights(self, wdict):
        for num, b in enumerate(self._binids):
            for hn, w in wdict.items():
                if hn in b:
                    self._W2[num] = w*w

    def __call__(self, x, nstart):
        self.setWeights({hn : _x for hn, _x in zip(self.hnames, x)})
        return self.minimize(nstart)
