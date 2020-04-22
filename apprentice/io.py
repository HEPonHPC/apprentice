
def read_limitsandfixed(fname):
    """
    Read a text file e.g.
    PARAM1  0         1   # interpreted as fixed param
    PARAM2  0.54444       # interpreted as limits
    """
    limits, fixed = {}, {}
    if fname is not None:
        with open(fname) as f:
            for l in f:
                if not l.startswith("#"):
                    temp = l.split()
                    if len(temp) == 2:
                        fixed[temp[0]] = float(temp[1])
                    elif len(temp) == 3:
                        limits[temp[0]] = (float(temp[1]), float(temp[2]))
    return limits, fixed

def readObs(fname):
    with open(fname) as f:
        r = [l.strip().split()[0] for l in f if not l.startswith("#")]
    return r

# def readH5(fname, idx=[0], xfield="params", yfield="values"):
    # """
    # Read X,Y values etc from HDF5 file.
    # By default, only the first object is read.
    # The X and Y-value dataset names depend on the file of course, so we allow
    # specifying what to use. yfield can be values|errors with the test files.
    # Returns a list of tuples of arrays : [ (X1, Y1), (X2, Y2), ...]
    # The X-arrays are n-dimensional, the Y-arrays are always 1D
    # """
    # import numpy as np
    # import h5py

    # with h5py.File(fname, "r") as f:
        # indexsize = f.get("index").size

    # # A bit of logic here --- if idx is passed an empty list, ALL data is read from file.
    # # Otherwise we need to check that we are not out of bounds.

    # # pnames = [p for p in f.get(xfield).attrs["names"]]
    # if len(idx) > 0:
        # assert (max(idx) <= indexsize)
    # else:
        # idx = [i for i in range(indexsize)]

    # ret = []
    # f = h5py.File(fname, "r")

    # # Read parameters
    # _X = np.array(f.get(xfield))

    # # Read y-values
    # for i in idx:
        # _Y = np.atleast_1d(f.get(yfield)[i])
        # USE = np.where((~np.isinf(_Y)) & (~np.isnan(_Y)))
        # ret.append([_X[USE], _Y[USE]])

    # f.close()

    # return ret


# # TODO rewrite such that yfield is a list of datasetnames, e.g. yfield=["values", "errors"]

# def readH52(fname, idx=[0], xfield="params", yfield1="values", yfield2="errors"):
    # """
    # Read X,Y, erros values etc from HDF5 file.
    # By default, only the first object is read.
    # The X and Y-value dataset names depend on the file of course, so we allow
    # specifying what to use. yfield can be values|errors with the test files.
    # Returns a list of tuples of arrays : [ (X1, Y1, E1), (X2, Y2, E2), ...]
    # The X-arrays are n-dimensional, the Y-arrays are always 1D
    # """
    # import numpy as np
    # import h5py

    # with h5py.File(fname, "r") as f:
        # indexsize = f.get("index").size

    # # A bit of logic here --- if idx is passed an empty list, ALL data is read from file.
    # # Otherwise we need to check that we are not out of bounds.

    # # pnames = [p for p in f.get(xfield).attrs["names"]]
    # if len(idx) > 0:
        # assert (max(idx) <= indexsize)
    # else:
        # idx = [i for i in range(indexsize)]

    # ret = []
    # f = h5py.File(fname, "r")

    # # Read parameters
    # _X = np.array(f.get(xfield))

    # # Read y-values
    # for i in idx:
        # _Y = np.atleast_1d(f.get(yfield1)[i])
        # _E = np.atleast_1d(f.get(yfield2)[i])
        # USE = np.where((~np.isinf(_Y)) & (~np.isnan(_Y)) & (~np.isinf(_E)) & (~np.isnan(_E)))
        # ret.append([_X[USE], _Y[USE], _E[USE]])

    # f.close()

    # return ret


def readH5(fname, idx, xfield="params", yfield1="values", yfield2="errors"):
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

    ret = []
    f = h5py.File(fname, "r")

    # Read parameters
    _X = np.array(f.get(xfield))
    Y = f.get(yfield1)[:][idx]

    if yfield2 in f:
        E = f.get(yfield2)[:][idx]
        # Read y-values
        for i in range(len(idx)):
            _Y = Y[i]
            _E = E[i]
            USE = np.where((~np.isinf(_Y)) & (~np.isnan(_Y)) & (~np.isinf(_E)) & (~np.isnan(_E)))
            ret.append([_X[USE], _Y[USE], _E[USE]])
    else:
        for i in range(len(idx)):
            _Y = Y[i]
            ret.append([_X, _Y, np.zeros(len(_Y))])

    f.close()
    return ret


def indexMapH5(fname, lsub):
    """
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f: II = [x.decode() for x in f.get("index")[:]]
    if len(lsub) == 0: lsub = np.unique([x.split("#")[0] for x in II])
    return {ls: np.where(np.char.find(II, ls) > -1)[0] for ls in lsub}


def readIndexH5(fname):
    import h5py
    with h5py.File(fname, "r") as f:
        return [x.decode() for x in f.get("index")[:]]


def readObsNamesH5(fname):
    import h5py
    with h5py.File(fname, "r") as f:
        return np.unique([x.decode().split("#")[0] for x in f.get("index")[:]])


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
    X = D[:, 0:-1]
    Y = D[:, -1]
    USE = np.where((~np.isinf(Y)) & (~np.isnan(Y)))
    return X[USE], Y[USE]

# NOTE deprecated
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


# Todo add binwidth in data model
def readExpData(fname, binids):
    import json
    import numpy as np
    with open(fname) as f: dd = json.load(f)
    Y = np.array([dd[b][0] for b in binids])
    E = np.array([dd[b][1] for b in binids])
    return dict([(b, (y, e)) for b, y, e in zip(binids, Y, E)])


def readTuneResult(fname):
    import json
    with open(fname) as f:
        return json.load(f)


def readApprox(fname, set_structures=True, usethese=None):
    import json, apprentice
    with open(fname) as f:
        rd = json.load(f)
    binids = sorted_nicely(rd.keys())
    binids = [x for x in binids if not x.startswith("__")]
    if usethese is not None:
        binids = [x for x in binids if x in usethese]

    APP = {}
    for b in binids:
        if "n" in rd[b]: APP[b] = apprentice.RationalApproximation(initDict=rd[b]) # FIXME what about set_structures for rationals?
        else:            APP[b] = apprentice.PolynomialApproximation(initDict=rd[b], set_structures=set_structures)
    return binids, [APP[b] for b in binids]
