#!/usr/bin/env python

__doc__="""

%prog [options] inputDir


Read in hierarchical mc run data just as usual and
convert into HDF5 file.

The following datasets are created:

    * runs  --- not really necessary
    * index --- NBIN bin names used to identify bits
    * params (with attribute names) --- the (NPARAMS * NRUNS) parameter array
    * values --- (NBIN * NRUNS) size array
    * errors --- (NBIN * NRUNS) size array
    * xmin   --- (NBIN * NRUNS) size array
    * xmax   --- (NBIN * NRUNS) size array
"""


def read_datafile(fname):
    with open(fname) as f: H=f.readline()
    colnames=H.strip().lstrip("#").strip().split()
    # We assume that the first two columns are bin edges
    D=np.loadtxt(fname)
    xmin = D[:,0]
    xmax = D[:,1]
    import os
    aname = os.path.basename(fname).split(".")[0]
    return {"/{}/{}".format(aname,col) : (xmin, xmax, D[:, colnames.index(col)]) for col in colnames[2:]}


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

def read_rundata(dirs, _infiles, _pfname="params.dat", verbose=True):
    """
    Read interpolation anchor point data from a provided set of run directory paths.
    """
    params, histos = {}, {}
    import os, glob
    numruns = len(dirs)
    for num, d in enumerate(sorted(dirs)):
        infiles = [os.path.join(d, _fin) for _fin in _infiles]
        pfname = os.path.join(d, _pfname)
        files = glob.glob(os.path.join(d, "*"))
        if not all([fin in files for fin in infiles]):
            if verbose: print("Skipping {} as some input files are missing".format(d))
            continue
        if not pfname in files:
            if verbose: print("Skipping {} as parameter file {} is  missing".format(d, pfname))
            continue

        pct = 100*(num+1)/float(numruns)
        if (num+1)%100 == 0: print("Reading run '%s' data: %d/%d = %2.0f%%" % (d, num+1, numruns, pct))

        params[d] = read_paramsfile(pfname)
        for fin in infiles:
            for path, hist in read_datafile(fin).items():
                histos.setdefault(path, {})[d] = hist

    return params, histos

import numpy as np
import h5py

if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTFILE", default="mc.hdf5", help="Output file name (default: %default)")
    op.add_option("-i", dest="INDIR", default=None, help="Input directory (default: %default)")
    op.add_option("-c", dest="COMPRESSION", type=int, default=4, help="GZip compression level (default: %default)")
    op.add_option("--pname", dest="PNAME", default="params.dat", help="Name of the params file to be found in each run directory (default: %default)")
    op.add_option("--log10", dest="LOG10", action='store_true', help="Store the log10 of values (default: %default)")
    opts, args = op.parse_args()

    print("NOTE: histograms are read in, converted to scaters. This has the effect that the nominal bin values are the bin heights. We multiply with the bin widths to correctly store the areas")



    ## Load MC run histos and params
    import glob
    INDIRS = glob.glob(os.path.join(opts.INDIR, "*"))
    INFILES = args

    PARAMS, HISTOS = read_rundata(INDIRS, INFILES, opts.PNAME, opts.DEBUG)

    # Parameter names and runs
    pnames = PARAMS[list(PARAMS.keys())[0]].keys()
    runs = sorted(list(PARAMS.keys()))


    # Iterate through all histos, bins and mc runs to rearrange data
    # in tables
    hbins ={}
    HNAMES=[str(x) for x in sorted(list(HISTOS.keys()))]
    BNAMES = []
    for hn in HNAMES:
        histos = HISTOS[hn]
        nbins = len(list(histos.values())[0][0])
        hbins[hn]=nbins
        for n in range(nbins):
            BNAMES.append("%s#%i"%(hn, n))

    vals = []
    xmin = []
    xmax = []
    for hn in HNAMES:
        for nb in range(hbins[hn]):
            try:
                vals.append([HISTOS[hn][r][-1][nb] if r in HISTOS[hn].keys() else np.nan for r in runs])
            except:
                from IPython import embed
                embed()
                exit(1)
            # Pick a run that actually exists here
            goodrun = runs[np.where(np.isfinite(vals[-1]))[0][0]]
            xmin.append(HISTOS[hn][goodrun][0][nb])
            xmax.append(HISTOS[hn][goodrun][1][nb])

    # Create new HDF5 file and write datasets
    f = h5py.File(opts.OUTFILE, "w")

    # https://github.com/h5py/h5py/issues/892
    f.create_dataset("runs",  data=np.char.encode(runs,   encoding='utf8'), compression=opts.COMPRESSION)
    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'),  compression=opts.COMPRESSION)
    pset = f.create_dataset("params", data=np.array([list(PARAMS[r].values()) for r in runs]), compression=9)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]

    if opts.LOG10:
        f.create_dataset("values", data=np.log10(vals), compression=opts.COMPRESSION)
    else:
        f.create_dataset("values", data=vals, compression=opts.COMPRESSION)
    f.create_dataset("xmin", data=xmin, compression=opts.COMPRESSION)
    f.create_dataset("xmax", data=xmax, compression=opts.COMPRESSION)
    f.close()

    print("Done. Output written to %s"%opts.OUTFILE)
    sys.exit(0)
