#!/usr/bin/env python

import h5py
import apprentice


if __name__ == "__main__":

    import os, sys
    if len(sys.argv)!=3:
        print("Usage: {} input.hf output.json".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    # Prevent overwriting of input data
    assert(sys.argv[2]!=sys.argv[1])

    # This reads only the first "bin's" information (for debugging)

    # DATA = apprentice.tools.readH5(sys.argv[1])


    # This reads the data for all bins

    # DATA = apprentice.tools.readH5(sys.argv[1], [])
    # idx = [i for i in range(len(DATA))]

    # This reads the data for a selection of bins
    idx = [0,1,2,5,7,20,44]
    DATA = apprentice.tools.readH5(sys.argv[1], idx)

    # Note: the idx is needed to make a connection to experimental data

    ras = []
    for X, Y in  DATA:
        ras.append(apprentice.RationalApproximation(X, Y, order=(3,1)))

    # This reads the unique identifiers of the bins
    with h5py.File(sys.argv[1], "r") as f:  binids = f.get("index")[idx]

    # jsonify
    JD = { str(x) : y.asDict for x, y in zip(binids, ras) }


    import json
    with open(sys.argv[2], "w") as f: json.dump(JD, f)

    print("Done --- approximation of {} objects written to {}".format(len(idx), sys.argv[2]))
