import json
import apprentice
import argparse
import h5py
import numpy as np
import os,sys
"""
python analyzeApprox.py -i ../../../log/SBNFIT/comparespectrum_mpi_deg2.h5 -a ../../../log/SBNFIT/approx/approximation_m2_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m3_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m3_n1_tCp.json ../../../log/SBNFIT/approx/approximation_m4_n1_tCp.json ../../../log/SBNFIT/approx/approximation_m10_n0_tCp.json ../../../log/SBNFIT/approx/approximation_m21_n1_tCp.json ../../../log/SBNFIT/approx/approximation_m22_n0_tCp.json
python analyzeApprox.py -i ../../../log/SBNFIT/comparespectrum_mpi_deg2.h5 -a  ../../../log/SBNFIT/approx/approximation_m4_n1_tCp.json -o ../../../log/SBNFIT/plots 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Approx analyzer for SBNFIT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--indfile", dest="INFILE", type=str, default=None,
                        help="Input H5 file")
    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default="/tmp",
                        help="Output directory")
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
    Xall = np.array(Xall)
    s=""
    signalvals = {}
    rbvals = {}
    errvals = {}
    for bno,bin in enumerate(binids):
        signalvals[bin] = {}
        rbvals[bin] = {}
        errvals[bin] = {}
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
            signalvals[bin][file] = Yall
            rbvals[bin][file] = vals
            errvals[bin][file] = err
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

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

    minval = np.Infinity
    maxval = -1*np.Infinity
    minerr = np.Infinity
    maxerr = -1*np.Infinity
    for bno, bin in enumerate(binids):
        for fno, file in enumerate(args.APPROXFILE):
            for valobj  in [signalvals[bin], rbvals[bin]]:
                # print("{} {} {}".format(bin,file,min(valobj[file])))
                minval = min([minval, min(valobj[file])])
                maxval = max([maxval, max(valobj[file])])
            minerr = min([minerr, min(errvals[bin][file])])
            maxerr = max([maxerr, max(errvals[bin][file])])

    for valtype in ["signal", "rb", "err"]:
        os.makedirs(os.path.join(args.OUTDIR,valtype),exist_ok=True)
    for bno, bin in enumerate(binids):
        for fno, file in enumerate(args.APPROXFILE):
            for valobj, valtype, vallab,minv,maxv in zip([signalvals[bin], rbvals[bin], errvals[bin]],
                                               ["signal", "rb", "err"],
                                               ["Signal", "Approximation", "Error"],
                                                [minval,minval,minerr],
                                                [maxval,maxval,maxerr]):
                df = pd.DataFrame()
                df['val'] = valobj[file]
                df['x'] = Xall[:, 0]
                df['y'] = Xall[:, 1]
                vv = np.array(valobj[file]).reshape(npointsperdim, npointsperdim)
                result = df.pivot(index='x', columns='y', values='val')
                # print(result)
                fig, ax = plt.subplots(figsize=(12, 7))
                title = "{} {} Heat Map".format(bin, vallab)
                plt.title(title,fontsize=18)
                ttl = ax.title
                ttl.set_position([0.5,1.05])

                ax.set_xticks([])
                ax.set_yticks([])

                ax.axis('off')
                # print(minv)
                # print(maxv)
                vvv = np.log10(valobj[file])
                sns.heatmap(vvv.reshape(npointsperdim,npointsperdim),
                            fmt="",cmap="RdYlGn",linewidths=0.3,ax=ax
                            # ,vmin=minv,vmax=maxv
                            )
                # plt.show()
                plt.savefig(os.path.join(args.OUTDIR,valtype,"{}_{}.pdf".format(bin,valtype)))
                plt.close()



    print(s)





















