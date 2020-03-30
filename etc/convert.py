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


def read_histos(path):
    "Load histograms from a YODA-supported file type, into a dict of path -> yoda.Histo[DataBin]"
    histos = {}
    try:
        import yoda
        s2s = []
        types = []
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
        print("Can't load histos from file '%s': %s" % (path, e))
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
    params, histos = {}, {}
    import os, glob, re
    re_pfname = re.compile(pfname) if pfname else None
    numruns = len(dirs)
    for num, d in enumerate(sorted(dirs)):
        pct = 100*(num+1)/float(numruns)
        if (num+1)%100 == 0: print("Reading run '%s' data: %d/%d = %2.0f%%" % (d, num+1, numruns, pct))
        files = glob.glob(os.path.join(d, "*"))
        for f in files:
            ## Params file
            if re_pfname and re_pfname.search(os.path.basename(f)):
                params[d] = read_paramsfile(f)
            else:
                if f.endswith("yoda"):
                    try:
                        ## Read as a path -> Histo dict
                        hs = read_histos(f)
                        ## Restructure into the path -> run -> Histo return dict
                        for path, hist in hs.items():
                            histos.setdefault(path, {})[d] = hist
                    except Exception as e:
                        print("Whoopsiedoodles {}".format(e))
                        pass #< skip files that can't be read as histos

        ## Check that a params file was found and read in this dir... or that no attempt was made to find one
        if pfname:
            if d not in params.keys():
                raise Exception("No params file '%s' found in run dir '%s'" % (pfname, d))
        else:
            params = None
    return params, histos

import numpy as np
import h5py

if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTFILE", default="mc.hdf5", help="Output file name (default: %default)")
    op.add_option("-c", dest="COMPRESSION", type=int, default=4, help="GZip compression level (default: %default)")
    op.add_option("--pname", dest="PNAME", default="params.dat", help="Name of the params file to be found in each run directory (default: %default)")
    op.add_option("--log10", dest="LOG10", action='store_true', help="Store the log10 of values (default: %default)")
    opts, args = op.parse_args()

    print("NOTE: histograms are read in, converted to scaters. This has the effect that the nominal bin values are the bin heights. We multiply with the bin widths to correctly store the areas")

    ## Import test
    try:
        import yoda
    except ImportError:
        raise Exception("YODA not found!")


    ## Load MC run histos and params
    import glob
    INDIRSLIST = [glob.glob(os.path.join(a, "*")) for a in args ]
    INDIRS     = [item for sublist in INDIRSLIST for item in sublist]

    PARAMS, HISTOS = read_rundata(INDIRS, opts.PNAME)

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
        nbins = len(list(histos.values())[0])
        hbins[hn]=nbins
        for n in range(nbins):
            BNAMES.append("%s#%i"%(hn, n))

    vals = []
    errs = []
    xmin = []
    xmax = []
    for hn in HNAMES:
        for nb in range(hbins[hn]):
            vals.append([HISTOS[hn][r][nb][2] if r in HISTOS[hn].keys() else np.nan for r in runs])
            errs.append([HISTOS[hn][r][nb][3] if r in HISTOS[hn].keys() else np.nan for r in runs])
            # Pick a run that actually exists here
            goodrun = runs[np.where(np.isfinite(vals[-1]))[0][0]]
            xmin.append(HISTOS[hn][goodrun][nb][0])
            xmax.append(HISTOS[hn][goodrun][nb][1])

    # Create new HDF5 file and write datasets
    f = h5py.File(opts.OUTFILE, "w")

    # https://github.com/h5py/h5py/issues/892
    f.create_dataset("runs",  data=np.char.encode(runs,   encoding='utf8'), compression=opts.COMPRESSION)
    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'),  compression=opts.COMPRESSION)
    pset = f.create_dataset("params", data=np.array([list(PARAMS[r].values()) for r in runs]), compression=9)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]

    if opts.LOG10:
        f.create_dataset("values", data=np.log10(vals), compression=opts.COMPRESSION)
        f.create_dataset("errors", data=np.log10(errs), compression=opts.COMPRESSION)
    else:
        f.create_dataset("values", data=vals, compression=opts.COMPRESSION)
        f.create_dataset("errors", data=errs, compression=opts.COMPRESSION)
    f.create_dataset("xmin", data=xmin, compression=opts.COMPRESSION)
    f.create_dataset("xmax", data=xmax, compression=opts.COMPRESSION)
    f.close()

    print("Done. Output written to %s"%opts.OUTFILE)
    sys.exit(0)
