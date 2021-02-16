import json
import apprentice
import argparse
import h5py
import numpy as np
"""
python analyzeApprox.py -i ../../../log/SBNFIT/comparespectrum_mpi_deg2.h5 -a ../../../log/SBNFIT/approx/approximation_m2_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m3_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m3_n1_tCp.json ../../../log/SBNFIT/approx/approximation_m4_n1_tCp.json ../../../log/SBNFIT/approx/approximation_m10_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m23_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m22_n1_tCp.json
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Approx analyzer for SBNFIT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    parser.add_argument("-a", "--approxfile", dest="APPROXFILE", default=[], nargs='+',
                        help="Input Approximation files")

    args = parser.parse_args()

    f = h5py.File(args.INFILE, "r")
    data = np.array(f.get('colspec'))
    f.close()

    nbin = np.shape(data)[1]
    npoints = np.shape(data)[0]
    npointsperdim = int(np.sqrt(npoints))
    binids = ["Bin{}".format(i) for i in range(nbin)]
    pnames = ['p1', 'p2']

    Xall = []
    for i in range(npointsperdim):
        for j in range(npointsperdim):
            Xall.append([i, j])
    s=""
    for bno,bin in enumerate(binids):
        Yall = data[:,bno]
        s+="\n\n{}".format(bin)
        s += "\nOrder\t\t\t\t"
        for fno, file in enumerate(args.APPROXFILE):
            app = apprentice.AppSet(file)
            sel = np.where(app._binids == bin)[0][0]
            if hasattr(app._RA[sel],'n'):
                s+= "({},{})\t\t".format(app._RA[sel].m,app._RA[sel].n)
            else:
                s += "({},{})\t\t".format(app._RA[sel].m, 0)
        s += "\nRange Extent\t\t"
        rearr = []
        maxerrarr = []
        ssqarr = []
        rmsarr = []
        for fno,file in enumerate(args.APPROXFILE):
            app = apprentice.AppSet(file)
            sel = np.where(app._binids == bin)[0][0]
            re = app._RA[sel].vmax - app._RA[sel].vmin
            rearr.append(re)
            s+="%.4E\t"%(re)
        s += "\nMax Error\t\t\t"
        for fno,file in enumerate(args.APPROXFILE):
            app = apprentice.AppSet(file)
            sel = np.where(app._binids == bin)[0][0]
            vals = [app.vals(p, sel=[sel]) for p in Xall]
            err = [np.abs(y - fb) for y, fb in zip(Yall, vals)]
            maxerr = max(err)
            maxerrarr.append(maxerr)
            ssq = np.sum([(y-fb)**2 for y, fb in zip(Yall, vals)])
            ssqarr.append(ssq)
            rmsarr.append(np.sqrt(ssq))
            s+= "%.4E\t" % (maxerr)
        s += "\nN. Max Error\t\t"
        for fno, file in enumerate(args.APPROXFILE):
            s += "%.4E\t" % (maxerrarr[fno]/rearr[fno])
        s += "\nSum of Sq\t\t\t"
        for fno, file in enumerate(args.APPROXFILE):
            s += "%.4E\t" % (ssqarr[fno])
        s += "\nRMS\t\t\t\t\t"
        for fno, file in enumerate(args.APPROXFILE):
            s += "%.4E\t" % (rmsarr[fno])
        s += "\nN. RMS\t\t\t\t"
        for fno, file in enumerate(args.APPROXFILE):
            s += "%.4E\t" % (rmsarr[fno]/rearr[fno])
        s+="\n\n"

    print(s)














