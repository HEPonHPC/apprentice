import sys

import apprentice as app
from apprentice.mpi4py_ import MPI_

def readInputDataH5(fname, wfile=None, comm=MPI_.COMM_WORLD):
    import apprentice as app
    import numpy as np
    import h5py
    # comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pnames, binids, IDX, xmin, xmax = None, None, None, None, None
    if rank==0:
        pnames      = app.io.readPnamesH5(fname, xfield="params")
        observables = list(set(app.io.readObs(wfile))) if wfile is not None else app.io.readObsNamesH5(fname)
        im          = app.io.indexMapH5(fname, observables)
        IDX         = np.sort(np.concatenate(list(im.values())))
        binids      = app.io.readIndexH5(fname)

        with h5py.File(fname, "r") as f:
            xmin = f["xmin"][:]
            xmax = f["xmax"][:]
    pnames  = comm.bcast(pnames     , root=0)
    binids  = comm.bcast(binids     , root=0)
    xmin    = comm.bcast(xmin, root=0)
    xmax    = comm.bcast(xmax, root=0)

    rankIdx = app.tools.chunkIt(IDX, size) if rank==0 else None
    rankIdx = comm.scatter(rankIdx, root=0)
    DATA    = app.io.readH5(fname, rankIdx)
    return DATA, np.array(binids)[rankIdx], pnames, rankIdx, xmin[rankIdx], xmax[rankIdx]

def readH5(fname, idx=None, xfield="params", yfield1="values", yfield2="errors"):
    """
    Read X,Y, errors values etc from HDF5 file.
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

        if idx is not None and len(idx) > 0:
            assert (max(idx) <= indexsize)
        else:
            idx = [i for i in range(indexsize)]
        ret = []
        # f = h5py.File(fname, "r")

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

def readSingleYODAFile(dirname, parFileName="params.dat", wfile=None):
    # if len(dirnames)>1: raise Exception("readSingleYODAFile cannot read more than one YODA run")
    import apprentice as app
    import numpy as np
    import yoda, glob, os
    # INDIRSLIST = [glob.glob(os.path.join(a, "*")) for a in dirnames]
    # indirs = [item for sublist in INDIRSLIST for item in sublist]
    PARAMS, HISTOS = app.io.read_rundata([dirname], parFileName)
    histos = []
    for k, v in HISTOS.items():
        temp = []
        for _k, _v in v.items():
            temp.append((_k, _v))
        histos.append((k, temp))
    _histos = {}
    _params = {}

    for p in PARAMS:_params.update(PARAMS[p])
    for rl in histos:
        for ih in range(0,len(rl),2):
            hname = rl[ih]
            if not hname in _histos: _histos[hname] = {}
            for ir in range(1, len(rl), 2):
                _histos[hname] = rl[ir][0][1]
    hbins = {}
    HNAMES = [str(x) for x in sorted(list(_histos.keys()))]
    if wfile is not None:
        observables = list(set(app.io.readObs(wfile)))
        HNAMES = [hn for hn in HNAMES if hn in observables]
    X = np.array(list(_params.values()))
    BNAMES = []
    for hn in HNAMES:
        histos = _histos[hn]
        nbins = len(list(histos))
        hbins[hn] = nbins
        for n in range(nbins):
            BNAMES.append("%s#%i" % (hn, n))
    _data, xmin, xmax = [], [], []
    for hn in HNAMES:
        for nb in range(hbins[hn]):
            vals = _histos[hn][nb][2]
            errs = _histos[hn][nb][3]
            _data.append([X, [np.array(vals)], [np.array(errs)]])
    return _data,BNAMES

def read_input_data_YODA_on_all_ranks(dirnames, parFileName="params.dat", wfile=None, storeAsH5=None,comm = MPI_.COMM_WORLD):
    import apprentice as app
    import numpy as np
    import yoda, glob, os
    rank = comm.Get_rank()
    indirs=None
    if rank==0:
        INDIRSLIST = [glob.glob(os.path.join(a, "*")) for a in dirnames]
        indirs     = [item for sublist in INDIRSLIST for item in sublist]
    indirs = comm.bcast(indirs, root=0)
    PARAMS, HISTOS = app.io.read_rundata(indirs, parFileName)
    send = []
    for k, v in HISTOS.items():
        temp = []
        for _k, _v in v.items():
            temp.append((_k, _v))
        send.append((k, temp))

    params = [PARAMS]
    histos = [send]

    _params = {}
    _histos = {}
    for p in params: _params.update(p)

    for rl in histos:
        for ih in range(len(rl)):
            hname = rl[ih][0]
            if not hname in _histos: _histos[hname] = {}
            for ir in range(len(rl[ih][1])):
                run =  rl[ih][1][ir][0]
                _histos[hname][run] = rl[ih][1][ir][1]
    pnames = [str(x) for x in _params[list(_params.keys())[0]].keys()]
    runs = sorted(list(_params.keys()))
    X=np.array([list(_params[r].values()) for r in runs])

    # Iterate through all histos, bins and mc runs to rearrange data
    hbins ={}
    HNAMES=[str(x) for x in sorted(list(_histos.keys()))]
    if wfile is not None:
        observables = list(set(app.io.readObs(wfile)))
        HNAMES = [hn for hn in HNAMES if hn in observables]
    BNAMES = []
    for hn in HNAMES:
        histos = _histos[hn]
        nbins = len(list(histos.values())[0])
        hbins[hn]=nbins
        for n in range(nbins):
            BNAMES.append("%s#%i"%(hn, n))

    _data, xmin, xmax = [], [], []
    for hn in HNAMES:
        for nb in range(hbins[hn]):
            vals = [_histos[hn][r][nb][2] if r in _histos[hn].keys() else np.nan for r in runs]
            errs = [_histos[hn][r][nb][3] if r in _histos[hn].keys() else np.nan for r in runs]
            # Pick a run that actually exists here
            isitfinite = (np.where(np.isfinite(vals))[0])
            if len(isitfinite) > 0:
                goodrun = runs[np.where(np.isfinite(vals))[0][0]]
                xmin.append(_histos[hn][goodrun][nb][0])
                xmax.append(_histos[hn][goodrun][nb][1])
            # USE = np.where((~np.isinf(vals)) & (~np.isnan(vals)) & (~np.isinf(errs)) & (~np.isnan(errs)))
            # Use everthing since nan will be handled by workflow
            USE = np.where((~np.isinf([i for i in range(len(vals))])))
            xg=X[USE,:]
            if len(xg.shape)==3:
                xg=xg.reshape(xg.shape[1:])
            _data.append([xg, np.array(vals)[USE], np.array(errs)[USE]])

    if storeAsH5 is not None:
        writeInputDataSetH5(storeAsH5, _data, runs, BNAMES, pnames, xmin, xmax)

    data = _data
    binids = BNAMES

    return data, binids, pnames, xmin, xmax

def readInputDataYODA(dirnames, parFileName="params.dat", wfile=None, storeAsH5=None, comm = MPI_.COMM_WORLD):
    import apprentice as app
    import numpy as np
    import yoda, glob, os
    size = comm.Get_size()
    rank = comm.Get_rank()

    indirs=None
    if rank==0:
        INDIRSLIST = [glob.glob(os.path.join(a, "*")) for a in dirnames]
        indirs     = [item for sublist in INDIRSLIST for item in sublist]
    indirs = comm.bcast(indirs, root=0)

    rankDirs = app.tools.chunkIt(indirs, size) if rank==0 else None
    rankDirs = comm.scatter(rankDirs, root=0)

    PARAMS, HISTOS = app.io.read_rundata(rankDirs, parFileName)
    send = []
    for k, v in HISTOS.items():
        temp = []
        for _k, _v in v.items():
            temp.append((_k, _v))
        send.append((k, temp))

    params = comm.gather(PARAMS, root=0)
    histos = comm.gather(send, root=0)


    rankIdx, binids, X, Y, E, xmin, xmax, pnames, BNAMES, data= None, None, None, None, None, None, None, None, None, None
    if rank==0:
        _params = {}
        _histos = {}
        for p in params: _params.update(p)

        for rl in histos:
            for ih in range(len(rl)):
                hname = rl[ih][0]
                if not hname in _histos: _histos[hname] = {}
                for ir in range(len(rl[ih][1])):
                    run =  rl[ih][1][ir][0]
                    _histos[hname][run] = rl[ih][1][ir][1]

        pnames = [str(x) for x in _params[list(_params.keys())[0]].keys()]
        runs = sorted(list(_params.keys()))
        X=np.array([list(_params[r].values()) for r in runs])

        # Iterate through all histos, bins and mc runs to rearrange data
        hbins ={}
        HNAMES=[str(x) for x in sorted(list(_histos.keys()))]
        if wfile is not None:
            observables = list(set(app.io.readObs(wfile)))
            HNAMES = [hn for hn in HNAMES if hn in observables]
        BNAMES = []
        for hn in HNAMES:
            histos = _histos[hn]
            nbins = len(list(histos.values())[0])
            hbins[hn]=nbins
            for n in range(nbins):
                BNAMES.append("%s#%i"%(hn, n))

        _data, xmin, xmax = [], [], []
        for hn in HNAMES:
            for nb in range(hbins[hn]):
                vals = [_histos[hn][r][nb][2] if r in _histos[hn].keys() else np.nan for r in runs]
                errs = [_histos[hn][r][nb][3] if r in _histos[hn].keys() else np.nan for r in runs]
                # Pick a run that actually exists here
                isitfinite = (np.where(np.isfinite(vals))[0])
                if len(isitfinite) > 0:
                    goodrun = runs[np.where(np.isfinite(vals))[0][0]]
                    xmin.append(_histos[hn][goodrun][nb][0])
                    xmax.append(_histos[hn][goodrun][nb][1])
                USE = np.where((~np.isinf(vals)) & (~np.isnan(vals)) & (~np.isinf(errs)) & (~np.isnan(errs)))
                xg=X[USE,:]
                if len(xg.shape)==3:
                    xg=xg.reshape(xg.shape[1:])
                _data.append([xg, np.array(vals)[USE], np.array(errs)[USE]])

        if storeAsH5 is not None:
            writeInputDataSetH5(storeAsH5, _data, runs, BNAMES, pnames, xmin, xmax)

        # TODO add weight file reading for obsevable filtering
        observables = np.unique([x.split("#")[0] for x in BNAMES])
        im   = {ls: np.where(np.char.find(BNAMES, ls) > -1)[0] for ls in observables}
        IDX  = np.sort(np.concatenate(list(im.values())))

        rankIdx = app.tools.chunkIt(IDX, size)
        data = app.tools.chunkIt(_data, size)
        xmin = app.tools.chunkIt(xmin, size)
        xmax = app.tools.chunkIt(xmax, size)
        binids = app.tools.chunkIt(BNAMES, size)
    rankIdx = comm.scatter(rankIdx, root=0)
    data = comm.scatter(data, root=0)
    xmin = comm.scatter(xmin, root=0)
    xmax = comm.scatter(xmax, root=0)
    binids = comm.scatter(binids, root=0)
    pnames = comm.bcast(pnames, root=0)


    comm.barrier()

    return data, binids, pnames, rankIdx, xmin, xmax

def writeInputDataSetH5(fname, data, runs, BNAMES, pnames, xmin, xmax, compression=4):
    import h5py
    import numpy as np
    f = h5py.File(fname, "w")

    # TODO change encoding to fixed size ascii
    # https://github.com/h5py/h5py/issues/892
    f.create_dataset("runs",  data=np.char.encode(runs,   encoding='utf8'),  compression=compression)
    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'),  compression=compression)
    pset = f.create_dataset("params", data=data[0][0], compression=compression)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]

    f.create_dataset("values", data=np.array([d[1] for d in data]), compression=compression)
    f.create_dataset("errors", data=np.array([d[2] for d in data]), compression=compression)
    f.create_dataset("xmin", data=xmin, compression=compression)
    f.create_dataset("xmax", data=xmax, compression=compression)
    f.close()


def read_histos(path):
    """
    Load histograms from a YODA-supported file type, into a dict of path -> yoda.Histo[DataBin]
    """
    import yoda
    from packaging import version
    if version.Version(yoda.__version__.decode()) < version.parse("1.8.0"):
        return read_yoda_pre180(path)

    histos = {}
    s2s, types = [], []
    aos = yoda.read(path, asdict=False)
    try:
        for ao in aos:
            import os
            if os.path.basename(ao.path()).startswith("_"): continue
            if "/RAW/" in ao.path(): continue
            types.append(ao.type())
            s2s.append(ao.mkScatter())
        del aos
        for s2, tp in zip(s2s, types):
            if s2.dim()!=2: continue
            bins = [(p.xMin(), p.xMax(), p.y(), p.yErrAvg()) for p in s2.points()] # This stores the bin heights as y-values
            histos[s2.path()] = bins
        del s2s
    except Exception as e:
        print("read_histos --- Can't load histos from file '%s': %s" % (path, e))
    return histos

def read_yoda_pre180(path):
    """
    Load histograms from a YODA-supported file type, into a dict of path -> yoda.Histo[DataBin]
    This is for yoda versions < 1.8.0
    """
    histos = {}
    import yoda
    s2s = []
    types = []
    try:
        aos = yoda.read(path, asdict=False)
        for ao in aos:
            import os
            ## Skip the Rivet cross-section and event counter objects
            # TODO: Avoid Rivet-specific behaviour by try block handling & scatter.dim requirements
            if os.path.basename(ao.path).startswith("_"): continue
            if "/RAW/" in ao.path: continue
            ##
            types.append(ao.type)
            s2s.append(ao.mkScatter())
        del aos #< pro-active YODA memory clean-up
        for s2, tp in zip(s2s, types):# filter(lambda x:x.dim==2, s2s): # Filter for Scatter1D
            if s2.dim!=2: continue
            bins = [(p.xMin, p.xMax, p.y, p.yErrAvg) for p in s2.points] # This stores the bin heights as y-values
            histos[s2.path] = bins
        del s2s #< pro-active YODA memory clean-up
    except Exception as e:
        print("read_yoda_pre180 --- Can't load histos from file '%s': %s" % (path, e))
    return histos

def read_paramsfile(path):
    """
    Read a file with parameters stored as key, value pairs.
    """
    from collections import OrderedDict
    rtn = OrderedDict()
    with open(path, "r") as f:
        L = [l.strip() for l in f if not l.startswith("#")]
        for num, line in enumerate(L):
            parts = line.split()
            if len(parts) == 2:
                rtn[parts[0]] = float(parts[1])
            elif len(parts) == 1:
                rtn["PARAM%i" % num] = float(parts[0])
            else:
                raise Exception("Error in parameter input format")
    return rtn

def read_rundata(dirs, pfname="params.dat", verbosity=1):
    """
    Read interpolation anchor point data from a provided set of run directory paths.
    """
    import apprentice as app
    params, histos = {}, {}
    import os, glob, re
    re_pfname = re.compile(pfname) if pfname else None
    numruns = len(dirs)
    for num, d in enumerate(sorted(dirs)):
        pct = 100*(num+1)/float(numruns)
        # if (num+1)%100 == 0: print("Reading run '%s' data: %d/%d = %2.0f%%" % (d, num+1, numruns, pct))
        files = glob.glob(os.path.join(d, "*"))
        for f in files:
            ## Params file
            if re_pfname and re_pfname.search(os.path.basename(f)):
                params[d] = app.io.read_paramsfile(f)
            else:
                if f.endswith("yoda"):
                    try:
                        # Read as a path -> Histo dict
                        hs = app.io.read_histos(f)
                        # Restructure into the path -> run -> Histo return dict
                        for path, hist in hs.items():
                            histos.setdefault(path, {})[d] = hist
                    except Exception as e:
                        print("Whoopsiedoodles {}".format(e))
                        pass #< skip files that can't be read as histos

        # Check that a params file was found and read in this dir... or that no attempt was made to find one
        if pfname:
            if d not in params.keys():
                raise Exception("No params file '%s' found in run dir '%s'" % (pfname, d))
        else:
            params = None
    return params, histos

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
                if not l.startswith("#") and not len(l.strip())==0:
                    temp = l.split("#")[0].split()
                    if len(temp) == 2:
                        fixed[temp[0]] = float(temp[1])
                    elif len(temp) == 3:
                        limits[temp[0]] = (float(temp[1]), float(temp[2]))
    return limits, fixed

def readObs(fname):
    with open(fname) as f:
        r = [l.strip().split()[0].split("#")[0].split("@")[0] for l in f if not l.startswith("#")]
    return r

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
    import numpy as np
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

def yodaDir2Dict(dname):
    """
    Recursively find and read all files ending with '.yoad' from directory dname.
    """
    import apprentice as app
    import pathlib
    bindict = {}
    for f in pathlib.Path(dname).rglob('*.yoda'):
        histos = app.io.read_histos(str(f.resolve()))
        for refname in sorted(histos.keys()):
            bins = histos[refname]
            hname=refname.replace("/REF", "",1)
            for num, b in enumerate(bins):
                binid="{}#{}".format(hname, num)
                bindict[binid]=(b[2], b[3])
    return bindict

# TODO add binwidth in data model?
# TODO what to do in case of missing refdata --- be robust or be precise?
def readExpData(fin, binids):
    import numpy as np
    import json, os
    if os.path.isdir(fin):
        bindict = yodaDir2Dict(fin)
    else:
         with open(fin) as f:
             bindict = json.load(f)
    Y = np.array([bindict[b][0] for b in binids])
    E = np.array([bindict[b][1] for b in binids])
    return dict([(b, (y, e)) for b, y, e in zip(binids, Y, E)])


def readTuneResult(fname):
    import json
    with open(fname) as f:
        return json.load(f)

def readApprox(fname, set_structures=True, usethese=None):
    import json, apprentice
    with open(fname) as f:
        rd = json.load(f)
    binids = app.tools.sorted_nicely(rd.keys())
    binids = [x for x in binids if not x.startswith("__")]
    if usethese is not None:
        binids = [x for x in binids if x in usethese]

    APP = {}
    for b in binids:
        if "n" in rd[b]: APP[b] = apprentice.RationalApproximation(initDict=rd[b]) # FIXME what about set_structures for rationals?
        else:            APP[b] = apprentice.PolynomialApproximation(initDict=rd[b], set_structures=set_structures)
    return binids, [APP[b] for b in binids]
