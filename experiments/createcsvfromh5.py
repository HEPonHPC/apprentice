import apprentice as app
import optparse, os, sys, h5py
import numpy as np
from shutil import copyfile

if __name__ == "__main__":

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("--log", dest="ISLOG", action='store_true', default=False,
                  help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("-o", dest="OUTPUT", default=None, help="Output folder (default: %default)")
    op.add_option("-i", dest="INDIR", default=None, help="In directory (default: %default)")
    opts, args = op.parse_args()

    # import json
    # with open("../../log/orig/A14/out_NNPDF.json", 'r') as f:
    #     ds = json.load(f)
    #
    # header = ds['chain-0']['pnames_inner']
    # header.append('y')
    # boxin = "/Users/mkrishnamoorthy/Box/PhysicsData/HEP_Fermi/A14"
    # boxout = "/Users/mkrishnamoorthy/Box/PhysicsData/HEP_Fermi/A14-withheader"
    #
    # from os import listdir
    # from os.path import isfile, join
    # import csv
    #
    # onlyfiles = [f for f in listdir(boxin) if isfile(join(boxin, f))]
    #
    # for file in onlyfiles:
    #     filein = os.path.join(boxin,file)
    #     fileout = os.path.join(boxout,file)
    #
    #     with open(fileout, 'w', newline='') as outcsv:
    #         writer = csv.writer(outcsv)
    #         writer.writerow(header)
    #
    #         with open(filein, 'r', newline='') as incsv:
    #             reader = csv.reader(incsv)
    #             writer.writerows(row for row in reader)
    #
    #
    #
    # exit(0)


    # if os.path.exists(opts.OUTPUT):
    #     uk = '_all'
    #     if uk in opts.OUTPUT and opts.WEIGHTS is not None:
    #         print("found {}".format(opts.OUTPUT))
    #         if opts.WEIGHTS is not None:
    #             weights = list(set(app.tools.readObs(opts.WEIGHTS)))
    #             for i in range(len(weights)):
    #                 weights[i] = weights[i].replace("_",'')
    #                 weights[i] = weights[i].replace("/", '')
    #             outdir2 = opts.OUTPUT.split(uk)[0]
    #             os.makedirs(outdir2,exist_ok=True)
    #             for file in os.listdir(opts.OUTPUT):
    #                 if file.split('#')[0].replace('_','') in weights:
    #                     pathin = os.path.join(opts.OUTPUT,file)
    #                     pathout = os.path.join(outdir2,file)
    #                     copyfile(pathin, pathout)
    #         sys.exit(0)
    #



    os.makedirs(opts.OUTPUT, exist_ok=True)
    apprfile = os.path.join(opts.INDIR, "approximation.json")
    datafile = os.path.join(opts.INDIR, "experimental_data.json")
    wtfile = os.path.join(opts.INDIR, "weights")
    TO = app.appset.TuningObjective2(
        wtfile, datafile, apprfile,f_error=None,
        filter_envelope=False,
        filter_hypothesis=False,
        debug=False)
    h5binids = app.tools.readIndex(args[0])
    TObinids = TO._binids

    for num, b in enumerate(h5binids):
        if b in TObinids:
            DATA = app.tools.readH53(args[0], [num])
            _X = DATA[0][0]
            _Y = DATA[0][1]
            _E = DATA[0][2]
            USE = np.where((_Y > 0)) if opts.ISLOG else np.where((_E >= 0))
            X = _X[USE]

            Y = np.log10(_Y[USE]) if opts.ISLOG else _Y[USE]
            Y = np.atleast_2d(Y)

            E = np.log10(_E[USE]) if opts.ISLOG else _E[USE]
            E = np.atleast_2d(E)

            outfile = os.path.join(opts.OUTPUT,h5binids[num].replace('/','_'))
            np.savetxt(outfile, np.hstack((X, Y.T, E.T)), delimiter=",")





