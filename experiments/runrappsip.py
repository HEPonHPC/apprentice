import apprentice
import numpy as np


def runRASIP(X, Y, fndesc, m, n, ts, outfolder):
    import json
    rasip = apprentice.RationalApproximationSIP(
    							X,
    							Y,
    							m=m,
    							n=n,
    							trainingscale=ts,
    							strategy=0,
    							roboptstrategy = 'msbarontime',
    							filterpyomodebug = 2,
    							debugfolder=outfolder+"/log",
    							fnname=fndesc,
    							fitstrategy = 'filter',
    							localoptsolver = 'scipy',
            )
            # print("Test error FS {} RS {}: 1N:{} 2N:{} InfN:{}".format(fs, rs,
            #                 raNorm(rrr, X[i_test], Y[i_test],1),
            #                 np.sqrt(raNorm(rrr, X[i_test], Y[i_test],2)),
            #                 raNormInf(rrr, X[i_test], Y[i_test])))
            # print("Total Approximation time {}\n".format(rrr.fittime))

    outfile = "%s/out/%s_p%d_q%d_ts%s.json"%(outfolder,fndesc,m,n,ts)
    rasip.save(outfile)


if __name__ == "__main__":

    import os, sys
    if len(sys.argv)!=7:
        print("Usage: {} infile fndesc m n trainingscale outfolder".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    try:
        X,Y = apprentice.tools.readData(sys.argv[1])
    except:
        DATA = apprentice.tools.readH5(sys.argv[1], [0])
        X, Y = DATA[0]


    runRASIP(
        X,
        Y,
        fndesc=sys.argv[2],
        m=int(sys.argv[3]),
        n=int(sys.argv[4]),
        ts=sys.argv[5],
        outfolder=sys.argv[6]
    )
