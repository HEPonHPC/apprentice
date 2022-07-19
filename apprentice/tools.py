import numpy as np
from collections import OrderedDict

def refitPoly(p, sc):
    """
    Recalculate coefficients of polynomial p for domain
    of scaler sc.
    """
    NC = numCoeffsPoly(p.dim, p.m)
    X  = p._scaler.drawSamples(NC)
    A  = np.prod(np.power(sc.scale(X), p._struct_p[:, np.newaxis]), axis=2).T
    b  = p.predictArray(X)
    z  = np.linalg.solve(A,b)
    return z

def refitPolyAX(p, A, X):
    """
    Recalculate coefficients of polynomial p for domain
    of scaler sc.
    """
    b  = p.predictArray(X)
    return np.linalg.solve(A,b)
    # Minimally faster
    # import scipy
    # return scipy.linalg.lapack.dgesv(A,b)[2]

def regularise(app, threshold=1e-6):
    pc = np.zeros_like(app._pcoeff)
    for num, c in enumerate(app._pcoeff):
        if abs(c)>threshold: pc[num] = c
    app._pcoeff=pc

    if hasattr(app, "qcoeff"):
        qc = np.zeros_like(app._qcoeff)
        for num, c in enumerate(app._qcoeff):
            if abs(c)>threshold: qc[num] = c
        app._qcoeff=qc

def denomMinMS(rapp, multistart=10):
    box=rapp._scaler.box_scaled
    from scipy import optimize
    opt = [optimize.minimize(lambda x:rapp.denom(x), sp, bounds=box) for sp in rapp._scaler.drawSamples_scaled(multistart)]
    Y = [o["fun"] for o in opt]
    X = [o["x"]   for o in opt]
    return X[np.argmin(Y)]

def denomMaxMS(rapp, multistart=10):
    box=rapp._scaler.box_scaled
    from scipy import optimize
    opt = [optimize.minimize(lambda x:-rapp.denom(x), sp, bounds=box) for sp in rapp._scaler.drawSamples_scaled(multistart)]
    Y = [o["fun"] for o in opt]
    X = [o["x"]   for o in opt]
    return X[np.argmin(Y)]

def denomChangesSignMS(rapp, multistart=10):
    xmin = denomMinMS(rapp, multistart)
    xmax = denomMaxMS(rapp, multistart)
    bad  = rapp.denom(xmin) * rapp.denom(xmax) <0
    if bad: return True,  xmin, xmax
    else:   return False, xmin, xmax


def calcApprox(X, Y, order, pnames, mode= "sip", onbtol=-1, debug=False, testforPoles=100, ftol=1e-9, itslsqp=200):
    M, N = order
    import apprentice as app
    if N==0:
        _app = app.PolynomialApproximation(X, Y, order=M, pnames=pnames)
        hasPole=False
    else:
        if mode == "la":    _app = app.RationalApproximation(X, Y, order=(M,N), pnames=pnames, strategy=2)
        elif mode == "onb": _app = app.RationalApproximationONB(X, Y, order=(M,N), pnames=pnames, tol=onbtol, debug=debug)
        elif mode == "sip":
            try:
                _app = app.RationalApproximationSLSQP(X, Y, order=(M,N), pnames=pnames, debug=debug, ftol=ftol, itslsqp=itslsqp)
            except Exception as e:
                print("Exception:", e)
                return None, True
        elif mode == "lasip":
            try:
                _app = app.RationalApproximation(X, Y, order=(M,N), pnames=pnames, strategy=2, debug=debug)
            except Exception as e:
                print("Exception:", e)
                return None, True
            has_pole = denomChangesSignMS(_app, 100)[0]
            if has_pole:
                try:
                    _app = app.RationalApproximationSLSQP(X, Y, order=(M,N), pnames=pnames, debug=debug, ftol=ftol, itslsqp=itslsqp)
                except Exception as e:
                    print("Exception:", e)
                    return None, True
        else:
            raise Exception("Specified mode {} does not exist, choose la|onb|sip".format(mode))
        hasPole = denomChangesSignMS(_app, testforPoles)[0]

    return _app, hasPole

def getLHSsamples(dim,npoints,criterion,minarr,maxarr,seed=87236):
    from pyDOE import lhs
    import apprentice
    np.random.seed(seed)
    X = lhs(dim, samples=npoints, criterion=criterion)
    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
    return s.scaledPoints

def getdLHSsamples(dim,npoints,criterion,minarr,maxarr,seed=87236):
    import apprentice
    from pyDOE import lhs
    np.random.seed(seed)
    epsarr = []
    for d in range(dim):
        # epsarr.append((maxarr[d] - minarr[d])/10)
        epsarr.append(10 ** -6)

    facepoints = int(2 * numCoeffsRapp(dim - 1, [int(m), int(n)]))
    insidepoints = int(npoints - facepoints)
    Xmain = np.empty([0, dim])
    # Generate inside points
    minarrinside = []
    maxarrinside = []
    for d in range(dim):
        minarrinside.append(minarr[d] + epsarr[d])
        maxarrinside.append(maxarr[d] - epsarr[d])
    X = lhs(dim, samples=insidepoints, criterion=criterion)
    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarrinside, b=maxarrinside)
    X = s.scaledPoints
    Xmain = np.vstack((Xmain, X))

    # Generate face points
    perfacepoints = int(np.ceil(facepoints / (2 * dim)))
    index = 0
    for d in range(dim):
        for e in [minarr[d], maxarr[d]]:
            index += 1
            np.random.seed(seed + index * 100)
            X = lhs(dim, samples=perfacepoints, criterion=criterion)
            minarrface = np.empty(shape=dim, dtype=np.float64)
            maxarrface = np.empty(shape=dim, dtype=np.float64)
            for p in range(dim):
                if (p == d):
                    if e == maxarr[d]:
                        minarrface[p] = e - epsarr[d]
                        maxarrface[p] = e
                    else:
                        minarrface[p] = e
                        maxarrface[p] = e + epsarr[d]
                else:
                    minarrface[p] = minarr[p]
                    maxarrface[p] = maxarr[p]
            s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarrface, b=maxarrface)
            X = s.scaledPoints
            Xmain = np.vstack((Xmain, X))
    Xmain = np.unique(Xmain, axis=0)
    X = Xmain
    formatStr = "{0:0%db}" % (dim)
    for d in range(2 ** dim):
        binArr = [int(x) for x in formatStr.format(d)[0:]]
        val = []
        for i in range(dim):
            if (binArr[i] == 0):
                val.append(minarr[i])
            else:
                val.append(maxarr[i])
        X[d] = val
    return X
def getFirstOutLevelWithOption(option):
    outlevelDict = getOutlevelDict()
    arr = [int(i) for i in outlevelDict.keys()]
    sarr = np.sort(arr)
    for level in sarr:
        if option in outlevelDict[str(level)]:
            return level
    return -1

def getOutlevelDef(outlevel):
    return getOutlevelDict()[str(int(outlevel))]

def getStatusDef(status):
    statusDict = {
        0:"OK",
        1:"Norm of the projected gradient too small",
        2:"Max iterations reached",
        3:"Simulation budget depleted",
        4:"MC was successful on less than 1 or N_p parameters (error)",
        5:"Trust region radius is less than a certain bound",
        6:"Fidelity is at a maximum value for a specified number of iterations",
        7:"The usable vals and errs of a bin was less than what was needed for constructing a polynomial of order (1,0)/"
          "Too many vals or errs for this bin were either nan or infty"
    }
    return statusDict[status]

def getOutlevelDict():
    outlevelDict = {
        "0": ["Silent"],
        "10": ["1lineoutput"],
        "11": ["1lineoutput","PKp1"],
        "20": ["1lineoutput","PKp1","interpolationPoints"],
        "30": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue"],
        "40": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue","NormOfStep"],
        "50": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue","NormOfStep","All"]
    }
    return outlevelDict

def writePythiaFiles(proccardfilearr, pnames, points, outdir, fnamep="params.dat", fnameg="generator.cmd"):
    import os
    from shutil import copyfile
    def readProcessCard(fname):
        with open(fname) as f:
            L = [l.strip() for l in f]
        return L
    from os.path import join, exists
    if not exists(outdir):
        os.makedirs(outdir)
    for num, p in enumerate(points):
        npad = "{}".format(num).zfill(1+int(np.ceil(np.log10(len(points)))))
        outd = join(outdir, npad)
        if not exists(outd):
            import os
            os.makedirs(outd)

        outfparams = join(outd, fnamep)
        with open(outfparams, "w") as pf:
            for k, v in zip(pnames, p):
                pf.write("{name} {val:e}\n".format(name=k, val=v))

        paramstr = "\n"
        for k, v in zip(pnames, p):
            paramstr += "{name} = {val:e}\n".format(name=k, val=v)
        for proccardfile in proccardfilearr:
            if fnameg is None:
                outfgenerator = join(outd, os.path.basename(proccardfile))
            else:
                outfgenerator = join(outd, fnameg)
            copyfile(proccardfile, outfgenerator)
            with open(outfgenerator, "a") as pg:
                pg.write(paramstr)


def readMemoryMap():
    import json
    pyhenson = False
    try:
        import pyhenson as h
        memorymap = h.get("MemoryMap")
        pyhenson = True
    except:
        with open("memorymap.json", 'r') as f:
            ds = json.load(f)
            memorymap = np.array(ds["MemoryMap"])
    return (memorymap, pyhenson)

def writeMemoryMap(memoryMap, forceFileWrite=False):
    import json
    pyhenson = False
    try:
        import pyhenson as h
        h.add("MemoryMap", memoryMap)
        pyhenson = True
    except:
        if "All" in getOutlevelDef(getFromMemoryMap(memoryMap=memoryMap, key="outputlevel")):
            print("Standalone run detected. I will store data structures "
                "in files for communication between tasks")
        ds = {"MemoryMap": memoryMap.tolist()}
        with open("memorymap.json", 'w') as f:
            json.dump(ds, f, indent=4)
    if forceFileWrite:
        k = getFromMemoryMap(memoryMap=memoryMap, key="iterationNo")
        ds = {"MemoryMap": memoryMap.tolist()}
        with open("memorymap_k{}.json".format(k), 'w') as f:
            json.dump(ds, f, indent=4)
    return pyhenson

def getWorkflowMemoryMap(dim=2):
    dim = int(dim)
    keymap = {
        "dim":0,
        "tr_center":range(1,1+dim),
        "min_param_bounds": range(1 + dim, 1 + (2 * dim)),
        "max_param_bounds": range(1 + (2 * dim),1 + (3 * dim)),
        "tr_radius":1 + (3 * dim),
        "tr_maxradius": 2 + (3 * dim),
        "tr_sigma": 3 + (3 * dim),
        "tr_eta": 4 + (3 * dim),
        "tr_gradientCondition": 5 + (3 * dim),
        "tr_gradientNorm": 6 + (3 * dim),
        "N_p": 7 + (3 * dim),
        "theta": 8 + (3 * dim),
        "thetaprime": 9 + (3 * dim),
        "kappa": 10 + (3 * dim),
        "fidelity": 11 + (3 * dim),
        "usefixedfidelity": 12 + (3 * dim),
        "maxfidelity": 13 + (3 * dim),
        "N_s": 14 + (3 * dim),
        "max_iteration": 15 + (3 * dim),
        "min_gradientNorm": 16 + (3 * dim),
        "max_simulationBudget": 17 + (3 * dim),
        "simulationbudgetused": 18 + (3 * dim),
        "iterationNo": 19 + (3 * dim),
        "outputlevel": 20 + (3 * dim),
        "status":21 + (3 * dim),
        "param_names":22 + (3 * dim),
        "useYODAoutput":23 + (3 * dim),
        "max_fidelity_iteration":24 + (3 * dim),
        "no_iters_at_max_fidelity":25 + (3 * dim),
        "radius_at_which_max_fidelity_reached":26 + (3 * dim)
    }
    return keymap

def createWorkflowMemoryMap(dim=2):
    keymap = getWorkflowMemoryMap(dim)
    n = 0
    for k in keymap:
        try:
            for t in keymap[k]:
                n = max(n,t)
        except:
            n = max(n, keymap[k])
    memorymap = np.zeros(n+1,dtype=np.float)
    memorymap[keymap['dim']] = dim
    return memorymap

def putInMemoryMap(memoryMap, key, value):
    if key == "file":
        import json
        with open(value, 'r') as f:
            ds = json.load(f)

        keymap = getWorkflowMemoryMap(ds['dim'])
        memoryMap = createWorkflowMemoryMap(ds['dim'])
        memoryMap[keymap["tr_radius"]] = ds['tr']['radius']
        j = 0
        for i in keymap["tr_center"]:
            memoryMap[i] =ds['tr']['center'][j]
            j+=1
        pnameds = {"param_names":ds["param_names"]}
        import os
        with open(os.path.join("param_names.json"), 'w') as f:
            json.dump(pnameds,f,indent=4)
        memoryMap[keymap["tr_maxradius"]] = ds['tr']['maxradius']
        memoryMap[keymap["tr_sigma"]] = ds['tr']['sigma']
        memoryMap[keymap["tr_eta"]] = ds['tr']['eta']
        useYODAoutput = False
        if "useYODAoutput" in ds:
            useYODAoutput = ds["useYODAoutput"]
        putInMemoryMap(memoryMap,"useYODAoutput",useYODAoutput)
        useFixedFidelity = True
        if "usefixedfidelity" in ds: useFixedFidelity = ds["usefixedfidelity"]
        maxfidelity = ds["fidelity"] if useFixedFidelity else ds["maxfidelity"]
        putInMemoryMap(memoryMap,"usefixedfidelity",useFixedFidelity)
        putInMemoryMap(memoryMap,"maxfidelity",maxfidelity)
        j = 0
        if "param_bounds" in ds:
            param_bounds = np.array(ds['param_bounds'])
        else:
            param_bounds = []
            for d in range(ds['dim']):
                param_bounds.append([-1*np.Infinity,np.Infinity])
            param_bounds = np.array(param_bounds)
        for i in keymap['min_param_bounds']:
            memoryMap[i] = param_bounds[:,0][j]
            j+=1
        j = 0
        for i in keymap['max_param_bounds']:
            memoryMap[i] = param_bounds[:, 1][j]
            j += 1

        for k in ["N_p","dim","theta","thetaprime","fidelity","N_s",
                  "max_iteration","max_fidelity_iteration","min_gradientNorm","kappa",
                  "max_simulationBudget"]:
            memoryMap[keymap[k]] = ds[k]
        return memoryMap

    elif key=="tr_center":
        keymap = getWorkflowMemoryMap(memoryMap[0])
        j = 0
        for i in keymap["tr_center"]:
            memoryMap[i] = value[j]
            j += 1
    elif key=="simulationbudgetused":
        keymap = getWorkflowMemoryMap(memoryMap[0])
        memoryMap[keymap[key]] += value
    elif key in ["tr_gradientCondition","useYODAoutput","usefixedfidelity"]:
        keymap = getWorkflowMemoryMap(memoryMap[0])
        memoryMap[keymap[key]] = float(value)
    else:
        keymap = getWorkflowMemoryMap(memoryMap[0])
        memoryMap[keymap[key]] = value

def getFromMemoryMap(memoryMap, key):
    keymap = getWorkflowMemoryMap(memoryMap[0])
    if key in ["tr_center","min_param_bounds","max_param_bounds"]:
        arr = []
        for i in keymap[key]:
            arr.append(memoryMap[i])
        return arr
    elif key == "param_names":
        import json
        with open("param_names.json",'r') as f:
            ds = json.load(f)
        return ds["param_names"]
    elif key in ["outputlevel","iterationNo","dim","simulationbudgetused","max_iteration","max_fidelity_iteration","N_p","status"]:
        return int(memoryMap[keymap[key]])
    elif key in ["tr_gradientCondition","useYODAoutput","usefixedfidelity"]:
        return bool(memoryMap[keymap[key]])
    else:
        return memoryMap[keymap[key]]

def extreme(app, nsamples=1, nrestart=1, use_grad=False, mode="min"):
    PF = 1 if mode=="min" else -1
    if use_grad: jac=lambda x:PF*app.gradient(x)
    else: jac=None
    from scipy import optimize

    res = []
    for i in range(nrestart):
        P = app._scaler.drawSamples(nsamples)
        V = [PF*app.predict(p) for p in P]
        imin = V.index(min(V))
        pstart = P[imin]
        _fmin = optimize.minimize(lambda x:PF*app.predict(x), pstart, bounds=app._scaler.box, jac=jac, method="TNC")
        res.append(_fmin["fun"])
    return PF*min(res)


def neighbours(arr, karr):
    n = len(arr)
    asum, maxcount, maxstartindex, maxendindex, changestartto = 0, 0, 0, 0, 0

    for i in range(n):
        if maxcount + 1 < n and (asum + arr[i]) <= karr[maxcount + 1]:
            asum += arr[i]
            maxcount += 1
            maxstartindex = changestartto
            maxendindex = i
        elif asum != 0:
            asum = asum - arr[i - maxcount] + arr[i]
            changestartto = i - maxcount + 1
    if maxstartindex > 0 and arr[maxstartindex - 1] < arr[maxendindex]:
        maxendindex -= 1
        maxstartindex -= 1
    elif maxendindex < n - 1 and arr[maxstartindex] > arr[maxendindex + 1]:
        maxendindex += 1
        maxstartindex += 1
    return maxcount, maxstartindex, maxendindex


def pInBox(P, box):
    for i in range(len(P)):
        if P[i] < box[i][0]: return False
        if P[i] > box[i][1]: return False
    return True




import re


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Standard chi2. w are the squared weights, d is the differences and e are the 1/error^2 terms
def fast_chi(w, d, e):
    return np.sum(w * d * d * e)


def meanerror(w, d, e, nb):  # WW edited
    return (np.sum(w * d * d * e)) / nb

def score(d, e, nb, method):
    s = d*d*e - np.log(e)
    if method == "meanscore": return np.mean(s)
    else:                     return np.median(s)

def fast_grad(w, d, e, g):
    v = -2 * w * d * e # TODO check where the minus comes from
    return np.sum(g * v.reshape((v.shape[0], 1)), axis=0)

# from numba import jit
# @jit
def fast_grad2(w, d, E2, e, g, ge):
    errterm=1./(E2 + e*e)
    v1  = -2 * w * d     * errterm
    v2  = +2 * w * d * d * errterm*errterm * e   # NOTE massive thx to JT for the -
    return np.sum(g * v1.reshape((v1.shape[0], 1)), axis=0) - np.sum(ge * v2.reshape((v2.shape[0], 1)), axis=0)


def least_square(y_data, y_mod, sigma2, w):
    return w / sigma2 * (y_mod - y_data) ** 2


def least_squares(y_data, y_mod, sigma2, w, idxs):
    """
    Least squares calculation for problem at hand, i.e. length of individual arguments is number of total bins.
    :param y_data: Data.
    :param y_mod: Model evaluations at data locations.
    :param sigma2: Measurement variances.
    :param w: Weights.
    :param idxs: Indices corresponding to observables
    :return:
    """
    n_o = len(idxs)  # number of observables
    chi2 = np.zeros(n_o)
    V = 0.
    for i in range(n_o):
        i1 = idxs[i][0];
        i2 = idxs[i][-1]
        chi2[i] = np.sum(np.array(least_square(y_data[i1:i2], y_mod[i1:i2], sigma2[i1:i2], 1.)))
        V += w[i1] * chi2[i]  # weights are for all the bins the same in one observable, thus we just take the first one
    return V, chi2


def numNonZeroCoeff(app, threshold=1e-6):
    """
    Determine the number of non-zero coefficients for an approximation app.
    """
    n = 0
    for p in app._pcoeff:
        if abs(p) > threshold: n += 1

    if hasattr(app, '_qcoeff'):
        for q in app._qcoeff:
            if abs(q) > threshold: n += 1

    return n


def numNLPoly(dim, order):
    if order < 2:
        return 0
    else:
        return numCoeffsPoly(dim, order) - numCoeffsPoly(dim, 1)


def numNL(dim, order):
    """
    Number of non-linearities.
    """
    m, n = order
    if n == 0: return numNLPoly(dim, m)
    if m < 2 and n < 2:
        return 0
    elif m < 2 and n >= 2:
        return numCoeffsPoly(dim, n) - numCoeffsPoly(dim, 1)
    elif n < 2 and m >= 2:
        return numCoeffsPoly(dim, m) - numCoeffsPoly(dim, 1)
    else:
        return numCoeffsRapp(dim, order) - numCoeffsRapp(dim, (1, 1))


def numCoeffsPoly(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    ntok = 1
    r = min(order, dim)
    for i in range(r):
        ntok = ntok * (dim + order - i) / (i + 1)
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
    while comb(dim + omax + 1, omax + 1) + 1 <= N:  # The '+1' stands for a order 0 polynomial's dof
        omax += 1
    return omax




def gradientRecursionSlow(dim, struct, X, jacfac):
    DER = []
    for coord in range(dim):
        der = [0.]
        for s in struct[1:]:  # Start with the linear terms
            if s[coord] == 0:
                der.append(0.)
                continue
            term = 1.0
            for i in range(len(s)):
                if i == coord:
                    term *= s[i] * jacfac[i]
                    term *= X[i] ** (s[i] - 1)
                else:
                    term *= X[i] ** s[i]
            der.append(term)
        DER.append(der)
    return DER


def gradientRecursion(X, struct, jacfac):
    """
    X ... scaled point
    struct ... polynomial structure
    jacfac ... jacobian factor
    returns array suitbale for multiplication with coefficient vector
    """
    import numpy as np
    dim = len(X)
    REC = np.zeros((dim, len(struct)))
    _RR = np.power(X, struct)
    for coord in range(dim):
        nonzero = np.where(struct[:, coord] != 0)
        RR = np.copy(_RR[nonzero])
        RR[:, coord] = jacfac[coord] * struct[nonzero][:, coord] * np.power(X[coord], struct[nonzero][:, coord] - 1)
        REC[coord][nonzero] = np.prod(RR, axis=1)
    return REC

# @jit(forceobj=True, parallel=True)
def gradientRecursionFast(X, struct, jacfac, NNZ, sred):
    """
    X ... scaled point
    struct ... polynomial structure
    jacfac ... jacobian factor
    NNZ  ... list of np.where results
    sred ... reduced structure
    returns array suitable for multiplication with coefficient vector
    """
    # import numpy as np
    dim = len(X)
    REC = np.zeros((dim, len(struct)))
    _RR = np.power(X, struct)
    nelem = len(sred[0])

    W=[_RR[nz] for nz in NNZ]

    for coord, (RR, nz) in enumerate(zip(W,NNZ)):
        RR[:, coord] = jacfac[coord] * sred[coord] *_RR[:nelem, coord]
        REC[coord][nz] = np.prod(RR, axis=1)

    return REC

def getPolyGradient(coeff, X, dim=2, n=2):
    from apprentice import monomial
    import numpy as np
    struct_q = monomial.monomialStructure(dim, n)
    grad = np.zeros(dim, dtype=np.float64)

    for coord in range(dim):
        """
        Partial derivative w.r.t. coord
        """
        der = [0.]
        if dim == 1:
            for s in struct_q[1:]:  # Start with the linear terms
                der.append(s * X[0] ** (s - 1))
        else:
            for s in struct_q[1:]:  # Start with the linear terms
                if s[coord] == 0:
                    der.append(0.)
                    continue
                term = 1.0
                for i in range(len(s)):
                    # print(s[i])
                    if i == coord:
                        term *= s[i]
                        term *= X[i] ** (s[i] - 1)
                    else:
                        term *= X[i] ** s[i]
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
    while comb(dim + omax + 1, omax + 1) + 1 <= N:  # The '+1' stands for a order 0 polynomial's dof
        omax += 1

    combs = []
    for m in reversed(range(omax + 1)):
        for n in reversed(range(m + 1)):
            if comb(dim + m, m) + comb(dim + n, n) <= N:
                combs.append((m, n))

    if mirror:
        temp = [tuple(reversed(i)) for i in combs]
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
    if len(out) > num:
        out[-2].extend(out[-1])
        out = out[0:-1]

    if len(out) != num:
        raise Exception("something went wrong in chunkIt, the target size differs from the actual size")

    return out

def mkCov(yerrs):
    import numpy as np
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

def prediction2YODA(fvals, Peval, fout="predictions.yoda", ferrs=None, wfile=None):
    import apprentice as app
    vals = app.AppSet(fvals)
    errs = app.AppSet(ferrs) if ferrs is not None else None

    P = [Peval[x] for x in vals._SCLR.pnames] if type(Peval)==dict else Peval

    Y  = vals.vals(P)
    dY = errs.vals(P) if errs is not None else np.zeros_like(Y)

    hids=np.array([b.split("#")[0] for b in vals._binids])
    hnames = sorted(set(hids))
    observables = sorted([x for x in set(app.io.readObs(wfile)) if x in hnames]) if wfile is not None else hnames

    with open(fvals) as f:
        import json
        rd = json.load(f)
        xmin = np.array(rd["__xmin"])
        xmax = np.array(rd["__xmax"])

    DX = (xmax-xmin)*0.5
    X  = xmin + DX
    Y2D = []
    import yoda
    for obs in observables:
        idx = np.where(hids==obs)
        P2D = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], Y[idx], DX[idx], dY[idx])]
        Y2D.append(yoda.Scatter2D(P2D, obs, obs))
    yoda.write(Y2D, fout)

def envelope2YODA(fvals, fout_up="envelope_up.yoda", fout_dn="envelope_dn.yoda", wfile=None):
    import apprentice as app
    vals = app.AppSet(fvals)

    Yup = np.array([r.vmax for r in vals._RA])
    Ydn = np.array([r.vmin for r in vals._RA])
    dY = np.zeros_like(Yup)
    hids=np.array([b.split("#")[0] for b in vals._binids])
    hnames = sorted(set(hids))
    observables = sorted([x for x in set(app.io.readObs(wfile)) if x in hnames]) if wfile is not None else hnames

    with open(fvals) as f:
        import json
        rd = json.load(f)
        xmin = np.array(rd["__xmin"])
        xmax = np.array(rd["__xmax"])

    DX = (xmax-xmin)*0.5
    X  = xmin + DX
    Y2Dup, Y2Ddn = [], []
    import yoda
    for obs in observables:
        idx = np.where(hids==obs)
        P2Dup = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], Yup[idx], DX[idx], dY[idx])]
        Y2Dup.append(yoda.Scatter2D(P2Dup, obs, obs))
        P2Ddn = [yoda.Point2D(x,y,dx,dy) for x,y,dx,dy in zip(X[idx], Ydn[idx], DX[idx], dY[idx])]
        Y2Ddn.append(yoda.Scatter2D(P2Ddn, obs, obs))
    yoda.write(Y2Dup, fout_up)
    yoda.write(Y2Ddn, fout_dn)

class TuningObjective(object):

    def __init__(self, *args, **kwargs):
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        if type(args[0]) == str:
            self.mkFromFiles(*args, **kwargs)
        else:
            self.mkFromData(*args, **kwargs)

    def mkReduced(self, keep, **kwargs):
        RA = list(np.array(self._RA)[keep])
        Y = self._Y[keep]
        E = self._E[keep]
        binids = list(np.array(self._binids)[keep])
        W2 = self._W2[keep]
        return TuningObjective(RA, Y, E, W2, binids, **kwargs)

    def setReduced(self, keep):
        self._RA = list(np.array(self._RA)[keep])
        self._Y = self._Y[keep]
        self._E = self._E[keep]
        self._binids = list(np.array(self._binids)[keep])
        self._W2 = self._W2[keep]

    # @classmethod
    def mkFromData(cls, RA, Y, E, W2, binids, **kwargs):
        cls._RA = RA
        cls._Y = Y
        cls._E = E
        cls._W2 = W2
        cls._binids = binids

        cls.setAttributes(**kwargs)

    def mkFromFiles(self, f_weights, f_data, f_approx, **kwargs):
        cache_recursions = kwargs["cache_recursions"] if kwargs.get("cache_recursions") is not None else True
        import apprentice
        import numpy as np
        binids, RA = apprentice.io.readApprox(f_approx, set_structures=False)
        hnames = [b.split("#")[0] for b in binids]
        bnums = [int(b.split("#")[1]) for b in binids]

        # Initial weights
        weights = self.initWeights(f_weights, hnames, bnums)

        # Filter here to use only certain bins/histos
        dd = apprentice.io.readExpData(f_data, [str(b) for b in binids])
        Y = np.array([dd[b][0] for b in binids])
        E = np.array([dd[b][1] for b in binids])

        # Filter for wanted bins here and get rid of division by zero in case of 0 error which is undefined behaviour
        good = []
        for num, bid in enumerate(binids):
            if weights[num] > 0 and E[num] > 0:
                if cache_recursions and RA[0]._scaler != RA[num]._scaler:
                    if self._debug: print("Warning, dropping bin with id {} to guarantee caching works".format(bid))
                    continue
                good.append(num)
            else:
                if self._debug: print("Warning, dropping bin with id {} as its weight or error is 0. W = {}, E = {}".format(bid,weights[num],E[num]))


        # TODO This needs some re-engineering to allow fow multiple filterings
        self._RA = [RA[g] for g in good]
        self._binids = [binids[g] for g in good]
        self._E = E[good]
        self._Y = Y[good]
        self._W2 = np.array([w * w for w in np.array(weights)[good]])

        # Do envelope filtering by default
        if kwargs.get("filter_envelope") is not None and not kwargs["filter_envelope"]:
            pass
        else:
            envindices = self.envelope()
            removedbinindices = np.setdiff1d(range(len(self._binids)), envindices)
            if self._debug:
                print("\n Envelope Filter removed {} bins".format(len(removedbinindices)))
                for b in sorted(removedbinindices):
                    print("Removing binid {} as it was filtered out by ENVELOPE filter".format(self._binids[b]))
            self.setReduced(envindices)
        if self._debug:print("")
        # Do hypothesis filtering by default
        if kwargs.get("filter_hypothesis") is not None and not kwargs["filter_hypothesis"]:
            pass
        else:
            self.setAttributes(**kwargs)
            hypoindices = self.hypofilt(0.05)
            removedbinindices = np.setdiff1d(range(len(self._binids)), hypoindices)
            if self._debug:
                print("\n Hypothesis Filter removed {} bins".format(len(removedbinindices)))
                for b in sorted(removedbinindices):
                    print("Removing binid {} as it was filtered out by HYPOTHESIS filter".format(self._binids[b]))
            self.setReduced(hypoindices)

        if (len(self._RA) == 0):
            print("Filtering removed all bins. Exiting now from Tuning Objective")
            import sys
            sys.exit(1)
        self.setAttributes(**kwargs)

    def setAttributes(self, **kwargs):
        noiseexp = int(kwargs.get("noise_exponent")) if kwargs.get("noise_exponent") is not None else 2
        self._dim = self._RA[0].dim
        self._E2 = np.array([1. / e ** noiseexp for e in self._E])
        self._SCLR = self._RA[0]._scaler  # Here we quietly assume already that all scalers are identical
        self._hnames = sorted(list(set([b.split("#")[0] for b in self._binids])))
        self._bounds = self._SCLR.box
        if kwargs.get("limits") is not None: self.setLimits(kwargs["limits"])
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        hdict, _ = history_dict(self._binids, self._hnames)
        self._hdict = hdict
        self._wdict = weights_dict(self._W2, self._hdict)
        self._idxs = indices(self._hnames, self._hdict)
        self._windex = []
        for inum, i in enumerate(self._idxs):
            for j in range(i[0], i[1]):
                self._windex.append(inum)

        cache_recursions = kwargs["cache_recursions"] if kwargs.get("cache_recursions") is not None else True

        if cache_recursions:
            # print("Congrats, you are using an experimental feature.")
            self.use_cache = True
            self.prepareCache()

            # need maximum extends of coefficients
            nmax_p=np.max([r._pcoeff.shape[0]                           for r in self._RA])
            nmax_q=np.max([r._qcoeff.shape[0] if hasattr(r, "n") else 0 for r in self._RA])
            nmax = max(nmax_p, nmax_q)

            # self._PC = np.zeros((len(self._RA), np.max([r._pcoeff.shape[0] for r in self._RA])))
            self._PC = np.zeros((len(self._RA), nmax))
            for num, r in enumerate(self._RA): self._PC[num][:r._pcoeff.shape[0]] = r._pcoeff

            # Denominator
            # nmax = np.max([r._qcoeff.shape[0] if hasattr(r, "n") else 0 for r in self._RA])
            if nmax_q > 0:
                self._hasRationals = True
                self._QC = np.zeros((len(self._RA), nmax))
                for num, r in enumerate(self._RA):
                    if hasattr(r, "n"):
                        self._QC[num][:r._qcoeff.shape[0]] = r._qcoeff
                    else:
                        self._QC[num][0] = None
                self._mask = np.where(np.isfinite(self._QC[:, 0]))
            else:
                self._hasRationals = False

        else:
            self.use_cache = False
            for r in self._RA:
                r.setStructures()

    def setAppStructures(self):
        for r in self._RA: r.setStructures()

    def prepareCache(self):
        import apprentice
        orders = []
        for r in self._RA:
            orders.append(r.m)
            if hasattr(r, "n"):
                orders.append(r.n)

        omax = max(orders)
        self._structure = apprentice.monomialStructure(self.dim, omax)
        # Gradient helpers
        self._NNZ  = [np.where(self._structure[:, coord] != 0) for coord in range(self.dim)]
        self._sred = np.array([self._structure[nz][:,num] for num, nz in enumerate(self._NNZ)])
        if self.dim == 1:
            self.recurrence = apprentice.monomial.recurrence1D
        else:
            self.recurrence = apprentice.monomial.recurrence

    def setCache(self, x):
        import apprentice
        xs = self._SCLR.scale(x)
        self._maxrec = self.recurrence(xs, self._structure)

    def scalersIdentical(self):
        """
        Sanity check to test if caching is possible
        """
        s = self._RA[0]._scaler
        for r in self._RA[1:]:
            if s != r._scaler:
                return False
        return True

    def setLimits(self, fname):
        lim, fix = read_limitsandfixed(fname)
        for num, pn in enumerate(self.pnames):
            if pn in lim:
                self._bounds[num] = lim[pn]

    def initWeights(self, fname, hnames, bnums):
        import apprentice
        matchers = apprentice.weights.read_pointmatchers(fname)
        weights = []
        for hn, bnum in zip(hnames, bnums):
            pathmatch_matchers = [(m, wstr) for m, wstr in matchers.items() if m.match_path(hn)]
            posmatch_matchers = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
            weights.append(w)
        return weights

    def setWeights(self, wdict,wexp=2):
        """
        Convenience function to update the bins weights.
        """
        if type(wdict) == OrderedDict:
            # self._wdict = {k:[] for k in wdict.keys()}
            self._wdict = OrderedDict([(k, []) for k in wdict.keys()])
            for num, b in enumerate(self._binids):
                for hn, w in wdict.items():
                    if hn in b:
                        self._W2[num] = w ** wexp
                        self._wdict[hn].append(w)
        else:
            # wdict2 = {hn: _x for hn, _x in zip(self.hnames, wdict)}
            wdict2 = OrderedDict([(hn, _x) for hn, _x in zip(self.hnames, wdict)])
            self.setWeights(wdict2,wexp)

    def envelope(self, nmultistart=10, sel=None):
        if hasattr(self._RA[0], 'vmin') and hasattr(self._RA[0], "vmax"):
            if self._RA[0].vmin is None or self._RA[0].vmax is None:
                return np.where(self._Y)  # use everything

            VMIN = np.array([r.vmin for r in self._RA])
            VMAX = np.array([r.vmax for r in self._RA])
            return np.where(np.logical_and(VMAX > self._Y, VMIN < self._Y))
        else:
            return np.where(self._Y)  # use everything

    def hypofilt(self, alpha, nstart=20, nrestart=10):
        keepids = []
        for hn in self._hnames:
            sel = self.obsBins(hn)
            res = self.minimize(nstart=nstart, nrestart=nrestart, sel=sel)
            param = res['x']
            rbvals = self.calc_f_val(param, sel=sel)
            chi2_test_arr = (rbvals - self._Y[sel]) ** 2 * self._E2[sel]
            chi2_test = sum(chi2_test_arr)
            # if chi2_test!=res["fun"]: print("Warning, chi2 calc is fishy: {} vs. {}".format(chi2_test, res["fun"]))

            npars, nbins = len(param), len(sel)
            # https://stackoverflow.com/questions/32301698/how-to-build-a-chi-square-distribution-table
            from scipy.stats import chi2
            chi2_critical = chi2.isf(alpha, nbins - npars)

            if chi2_test > chi2_critical:
                chi2_critical_arr = np.zeros(nbins)
                chi2_critical_arr[:npars + 1] = np.inf
                chi2_critical_arr[npars + 1:] = chi2.isf(alpha, np.arange(1, nbins - npars))
                bcount, bstart, bend = neighbours(chi2_test_arr, chi2_critical_arr)
                # TODO: Check this special case
                if bcount < npars + 1:
                    if np.sum(chi2_test_arr[bstart:bend + 1]) > chi2_critical:
                        bcount, bstart, bend = 0, -1, -1
                    else:
                        chi2_critical_arr = [chi2_critical] * len(sel)
                        bcount, bstart, bend = neighbours(chi2_test_arr, chi2_critical_arr)

                if bcount == 0: continue
                for ikeep in range(bstart, bend + 1):
                    keepids.append(self._binids[sel[ikeep]])
                # nnn = len(range(bstart, bend + 1))
                # if nnn < len(sel):
                #     print("%s & %d & %.2f & %.2f & %.2f\\\\\\hline"%(hn.replace('_','\\_'),len(sel)-nnn,chi2_critical,chi2_test,np.sum(chi2_test_arr[bstart:bend + 1])))
            else:
                for ikeep in range(len(sel)): keepids.append(self._binids[sel[ikeep]])
        return [self._binids.index(x) for x in keepids]

    def fmin(self, nmultistart=10, sel=None):
        return [(i % 10 == 0 and print(i)) or r.fmin(nmultistart) for i, r in enumerate(self._RA)] if sel is None else [
            self._RA[num].fmin(nmultistart) for num in sel]

    def fmax(self, nmultistart=10, sel=None):
        return [(i % 10 == 0 and print(i)) or r.fmax(nmultistart) for i, r in enumerate(self._RA)] if sel is None else [
            self._RA[num].fmax(nmultistart) for num in sel]

    def weights_obs(self):
        """
        Get weights for individual observables.
        :return: Weights of first bin in each observable.
        """
        return [w[0] for w in self._wdict.values()]

    def obsGoF(self, hname, x, method):
        """
        Convenience function to get the (unweighted) contribution to the gof for obs hname at point x
        """
        import numpy as np
        _bids = [num for num, b in enumerate(self._binids) if hname in b]
        RR = [self._RA[i] for i in _bids]
        vals = [r(x) for r in RR]
        #        return fast_chi(np.ones(len(_bids)), self._Y[_bids], [self._RA[i](x) for i in _bids], self._E2[_bids] , len(_bids))
        # WW commented
        if method == "portfolio":
            return meanerror(np.ones(len(_bids)), self._Y[_bids] - vals, self._E2[_bids], len(_bids))  # WW edited
        else:  # scoring method
            return score(self._Y[_bids] - vals, self._E2[_bids], len(_bids), method)  # WW edited

    def meanCont(self, x, method):
        """
        Convenience function that return a list if the obsGof for all hnames at x
        """
        return [self.obsGoF(hn, x, method) for hn in self.hnames]

    @property
    def hnames(self):
        return self._hnames

    @property
    def dim(self):
        return self._dim

    @property
    def pnames(self):
        return self._SCLR.pnames

    def _objective_obs(self, x):
        """
        Return objective and individual contributions of observables.
        :param x:
        :return:
        """
        return least_squares(self._Y, [f(x) for f in self._RA], 1 / self._E2, np.sqrt(self._W2),
                             self._idxs)  # E2 is reciprocal


    def getVals(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.setCache(x)
        vals = np.sum(self._maxrec * self._PC[sel], axis=1)
        if self._hasRationals:
            den = np.sum(self._maxrec * self._QC[sel], axis=1)
            vals[self._mask[sel]] /= den[self._mask[sel]]
        return vals

    def getGrads(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.setCache(x)
        xs = self._SCLR.scale(x)
        JF = self._SCLR.jacfac
        GREC = gradientRecursionFast(xs, self._structure, self._SCLR.jacfac, self._NNZ, self._sred)

        Pprime = np.sum(self._PC[sel].reshape((self._PC[sel].shape[0], 1, self._PC[sel].shape[1])) * GREC, axis=2)

        if self._hasRationals:
            if set_cache: self.setCache(x)
            P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            Qprime = np.sum(self._QC[sel].reshape((self._QC[sel].shape[0], 1, self._QC[sel].shape[1])) * GREC, axis=2)
            return Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime

        return Pprime


    def objective(self, x, sel=slice(None, None, None), unbiased=False):
        if not self.use_cache:
            if isinstance(sel, list) or type(sel).__module__ == np.__name__:
                RR = [self._RA[i] for i in sel]
                vals = [r(x) for r in RR]
            #                vals = [r(x) for r in self._RA[sel]] #--> results in bug
            else:
                RR = self._RA[sel]
                vals = [f(x) for f in RR]
        else:
            self.setCache(x)
            vals = np.sum(self._maxrec * self._PC[sel], axis=1)
            if self._hasRationals:
                den = np.sum(self._maxrec * self._QC[sel], axis=1)
                vals /= den

        if unbiased:
            return fast_chi(np.ones(len(vals)), self._Y[sel] - vals, self._E2[sel])
        else:
            return fast_chi(self._W2[sel], self._Y[sel] - vals, self._E2[sel])

    def gradient(self, x, sel=slice(None, None, None), unbiased=False):
        self.setCache(x)
        # vals = np.sum(self._maxrec * self._PC[sel], axis=1)
        vals  = self.getVals( x, sel, set_cache=False)
        grads = self.getGrads(x, sel, set_cache=False)
        # X = self._SCLR.scale(x)
        # JF = self._SCLR.jacfac
        # struct = self._structure
        # GR = gradientRecursion(X, struct, JF)
        # temp = np.sum(self._PC.reshape((self._PC.shape[0], 1, self._PC.shape[1])) * GR, axis=2)

        return fast_grad(self._W2[sel], self._Y[sel] - vals, self._E2[sel], grads)

    def calc_f_val(self, x, sel=slice(None, None, None)):
        import autograd.numpy as np
        if not self.use_cache:
            if isinstance(sel, list) or type(sel).__module__ == np.__name__:
                RR = [self._RA[i] for i in sel]
                vals = [r(x) for r in RR]
                # vals = [self._RA[i](x) for i in sel] --> results in bug
            else:
                RR = self._RA[sel]
                vals = [f(x) for f in RR]
        else:
            self.setCache(x)
            vals = np.sum(self._maxrec * self._PC[sel], axis=1)
            if self._hasRationals:
                den = np.sum(self._maxrec * self._QC[sel], axis=1)
                vals /= den
        return vals

    def obsBins(self, hname):
        return [i for i, item in enumerate(self._binids) if item.startswith(hname)]

    def obswiseObjective(self, x, unbiased=False, binids=None, setCache=True):
        if binids is None:
            return [self.objective(x, sel=self.obsBins(hn), unbiased=unbiased) for hn in self._hnames]
        else:
            return [self.objective(x, sel=bids, unbiased=unbiased) for bids in binids]


    def XisbetterthanY(self, x, y):
        lchix = self.obswiseObjective(x, unbiased=True)
        lchiy = self.obswiseObjective(y, unbiased=True)

        comp = [lx < ly for lx, ly in zip(lchix, lchiy)]
        return comp.count(True) > comp.count(False)

    def startPoint(self, ntrials):
        if ntrials == 0:
            if self._debug: print("StartPoint: {}".format(self._SCLR.center))
            return self._SCLR.center
        import numpy as np
        _PP = np.random.uniform(low=self._SCLR._Xmin, high=self._SCLR._Xmax, size=(ntrials, self._SCLR.dim))
        _CH = [self.objective(p) for p in _PP]
        if self._debug: print("StartPoint: {}".format(_PP[_CH.index(min(_CH))]))
        return _PP[_CH.index(min(_CH))]

    def minimize(self, nstart, nrestart=1, sel=slice(None, None, None), use_grad=False, method="L-BFGS-B"):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        for t in range(nrestart):
            if use_grad:
                if self._debug: print("using gradient")
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), self.startPoint(nstart),
                                        bounds=self._bounds, jac=self.gradient, method=method)
            else:
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), self.startPoint(nstart),
                                        bounds=self._bounds, method=method)
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        return finalres

    def minimize_mpi(self, nstart, nrestart=10, sel=slice(None, None, None)):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        import apprentice as app
        rankWork = app.tools.chunkIt([_ for _ in range(nrestart)], comm.Get_size()) if rank == 0 else []
        rankWork = comm.scatter(rankWork, root=0)
        np.random.seed(rank)

        from scipy import optimize
        R = [optimize.minimize(lambda x: self.objective(x, sel=sel), self.startPoint(nstart), bounds=self._bounds) for _
             in rankWork]
        X = [r.x.tolist() for r in R]
        FUN = [r.fun.tolist() for r in R]
        ibest = np.argmin(FUN)
        X = comm.gather(X[ibest], root=0)
        FUN = comm.gather(FUN[ibest], root=0)

        xbest, fbest = None, None
        if rank == 0:
            ibest = np.argmin(FUN)
            xbest = X[ibest]
            fbest = FUN[ibest]
            comm.bcast(xbest, root=0)
            comm.bcast(fbest, root=0)

        return xbest, fbest

    def __len__(self):
        return len(self._binids)

    def __call__(self, x):
        return self.objective(x)


def history_dict(binids, hnames=None):
    if hnames is None:
        hnames = [b.split("#")[0] for b in binids]
    hdict = OrderedDict([(h, []) for h in hnames])
    for b in binids:
        hname, bid = b.split("#")
        hdict[hname].append(bid)
    return hdict, hnames


def weights_dict(w2_bins, hdict):
    """
    Transform bin weights to weights dictionary.
    :param w2_bins: Squared weights of bins.
    :param hdict: History dictionary.
    :return: Weights dictionary.
    """
    # wdict = {k:np.zeros(len(v)) for (k,v) in zip(hdict.keys(),hdict.values())}
    wdict = OrderedDict([(k, np.zeros(len(v))) for (k, v) in zip(hdict.keys(), hdict.values())])
    i = 0
    for (k, v) in zip(wdict.keys(), wdict.values()):
        n = len(v)
        wdict[k][:] = np.sqrt(np.array(w2_bins[i:i + n]))
        i += n
    return wdict


def indices(hnames, dict):
    """
    Returns indices with indices corresponding to observables.
    :param hnames: Names of observables. This is important as this determines the order of the observables! The dictionaries of the object, i.e. hdict and wdict, might be unordered. -> TODO: Use OrderedDict() instead?
    :param dict: Either weights or history dict.
    :return: Indices dictionary.
    """
    idxs = [[0, 0] for _ in hnames]
    i = 0
    for (k, v) in zip(range(len(hnames)), dict.values()):
        n = len(v)
        idxs[k][:] = [i, i + n]
        i += n
    return idxs


def artificial_data_from_RA(approximation_file, p0, eps=None, var=None, outfile=None, model_bias=None):
    """
    Create in-silico data from rational approximation file corrupting the data with zero mean Gaussian noise (lenghts of provided arguments must match the number of observables in the provided approximation file).
    :param approximation_file: Approximation json file.
    :param p0: True parameter value.
    :param eps: Variance factor (multiplied by the data value at the corresponding bin).
    :param var: Variances of measurement errors.
    :param outfile: Output json file path.
    :param model_bias: Bias for model error (shifts the mean value).
    :return: Experimental data json file (data with corresponding standard deviation), expected values of data.
    """
    np.random.seed(456785)
    import json
    binids, RA = readApprox(approximation_file)
    hdict, _ = history_dict(binids)
    hnames = hdict.keys()
    n_o = len(hnames)
    if eps is None:
        eps = np.zeros(n_o)
    if var is None:
        var = np.zeros(n_o)
    if model_bias is None:
        model_bias = np.zeros(n_o)
    if n_o != len(p0) or n_o != len(eps) or n_o != len(var) or n_o != len(model_bias):
        raise TypeError(
            "Lengths of p0, eps, var and model_bias (if provided) and number of observables have to be the same. There are {} observables.".format(
                n_o))
    RA_dict = dict([(b, r) for (b, r) in zip(binids, RA)])
    data = dict([(b, []) for b in binids])

    Ey = []  # expected values of data

    for (h, p, e, v, b) in zip(hnames, p0, eps, var, model_bias):
        for i in hdict[h]:
            bid = "{h}#{i}".format(h=h, i=i)
            r = RA_dict[bid]
            mu = r(p)
            sigma2_eps = e * abs(mu)
            d = mu + b + np.random.normal(0.0, np.sqrt(sigma2_eps)) + np.random.normal(0.0, np.sqrt(v))
            data[bid] = [d, np.sqrt(sigma2_eps + v)] if sigma2_eps + v > 0 else [d, 1.0]
            Ey.append(mu + b)

    if outfile is None:
        outfile = 'data.json'
    with open(outfile, 'w') as f:
        json.dump(data, f)

    return Ey


def generate_data_from_RA(approximationfile, experimentaldatafile, p0, bbdict, restart_filter=100, seed=54321, N=100,
                          epsmodel=0.):
    np.random.seed(seed)
    import json
    with open(experimentaldatafile, 'r') as f:
        expdata = json.load(f)
    binids, RA = readApprox(approximationfile)
    hdict, _ = history_dict(binids)
    hnames = hdict.keys()

    if len(hnames) != len(p0):
        n_o = len(hnames)
        raise TypeError(
            "Lengths of p0 and number of observables have to be the same. There are {} observables.".format(n_o))
    if len(expdata.keys()) != len(binids) or len(bbdict.keys()) != len(binids):
        n_b = len(binids)
        raise TypeError(
            "Number of keys in experimental data, bad bin and number of bins have to be the same. There are {} bins.".format(
                n_b))

    RA_dict = dict([(b, r) for (b, r) in zip(binids, RA)])
    data = dict([(b, []) for b in binids])

    for obs, p in zip(hdict.keys(), p0):
        for b in hdict[obs]:
            bid = "{o}#{b}".format(o=obs, b=b)
            r = RA_dict[bid]

            sigma = expdata[bid][1]

            if bbdict[bid]:
                bool = int(np.random.uniform(0, 2))
                if bool:
                    FEX = r.fmin(restart_filter)
                    mult = -1.
                else:
                    FEX = r.fmax(restart_filter)
                    mult = 1.

                mu = np.random.uniform(FEX + (mult * 0.2 * FEX), FEX + (mult * 0.8 * FEX))
            else:
                mu = r(p)

            sigmamodel = epsmodel * abs(mu)
            d = [mu + np.random.normal(0.0, scale=sigmamodel) + np.random.normal(0.0, scale=sigma) for i in range(N)]
            d = np.average(d)

            data[bid] = [d, sigma]

    return data


if __name__ == "__main__":
    import os, sys

    # memorymap = putInMemoryMap(None,key='file',value="/Users/mkrishnamoorthy/Research/Code/log/DFO/P/X2_2D_1bin/algoparams_bk.json")
    # print(memorymap)
    # print(getFromMemoryMap(memorymap,'tr_center'))
    # print(getFromMemoryMap(memorymap, 'N_p'))
    # print(getFromMemoryMap(memorymap, 'max_param_bounds'))
    # print(getFromMemoryMap(memorymap, '/Users/mkrishnamoorthy/Research/Code/log/DFO/P/X2_2D_1bin/param_names.json'))
    # exit(1)

    approximationfile = "../../pyoo/test_data_min2_noisefree/approximation.json"
    experimentaldatafile = "../../pyoo/test_data_min2_noisefree/experimental_data.json"
    import json

    with open(experimentaldatafile, 'r') as f:
        expdata = json.load(f)
    p = [2.4743870765622695, 1.7068479984402454]
    hnames = [b.split("#")[0] for b in expdata]
    uniquehnames = np.unique(hnames)
    p0 = [p] * len(uniquehnames)
    bbdict = {}
    for ono, obs in enumerate(uniquehnames):
        for b in expdata:
            if obs in b:
                bbdict[b] = False

    bbdict = {b: False for b in expdata}
    data = generate_data_from_RA(approximationfile, experimentaldatafile, p0, bbdict)
    sys.exit(0)
