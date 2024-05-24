#!/usr/bin/env python3

import apprentice.io as IO
import apprentice.tools as TOOLS
import numpy as np
import time

if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-w", dest="WEIGHTS", default=False, action='store_true', help="print a weight file (default: %default)")
    opts, args = op.parse_args()

    APPROX = args[0]

    import json

    with open(APPROX) as f: rd = json.load(f)
    blows = rd["__xmin"]
    rd.pop('__xmin',None)
    bups  = rd["__xmax"]
    rd.pop('__xmax',None)
    if '__vmin' in rd.keys():
        vlows = rd["__vmin"]
        rd.pop('__vmin',None)
        vups  = rd["__vmax"]
        rd.pop('__vmax',None)

    binids = TOOLS.sorted_nicely( rd.keys() )

    if not opts.WEIGHTS:
        rk = binids[0]
        print(rd[rk]['fnspace'])
        sys.exit(0)

    hnames  = [    b.split("#")[0]  for b in binids]
    bnums   = [int(b.split("#")[1]) for b in binids]

    nbins = []
    for hn in hnames:
        nbins.append(sum([1 for b in binids if hn in b]))

    if opts.WEIGHTS:
        wdict = dict(zip(hnames,nbins))
        for hn in wdict.keys():
            nb = wdict[hn]
            print("{}\t1.0 # {} bins".format(hn,nb))

