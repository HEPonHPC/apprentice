import apprentice as app
import optparse, os, sys, h5py
import numpy as np


if __name__ == "__main__":

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("--log", dest="ISLOG", action='store_true', default=False,
                  help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("-o", dest="OUTPUT", default=None, help="Output folder (default: %default)")
    opts, args = op.parse_args()
    os.makedirs(opts.OUTPUT, exist_ok=True)
    binids = app.tools.readIndex(args[0])

    for num, b in enumerate(binids):
        DATA = app.tools.readH53(args[0], [num])
        _X = DATA[0][0]
        _Y = DATA[0][1]
        _E = DATA[0][2]
        USE = np.where((_Y > 0)) if opts.ISLOG else np.where((_E >= 0))
        X = _X[USE]
        Y = np.log10(_Y[USE]) if opts.ISLOG else _Y[USE]
        Y = np.atleast_2d(Y)
        outfile = os.path.join(opts.OUTPUT,binids[num].replace('/','_'))
        np.savetxt(outfile, np.hstack((X, Y.T)), delimiter=",")





