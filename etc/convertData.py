#!/usr/bin/env python

import professor2 as prof



if __name__ == "__main__":
    import sys
    REF = sys.argv[1]

    DATA = prof.read_all_histos(REF)

    ddict = {}

    for hname in sorted(DATA.keys()):
        H = DATA[hname]
        for b in range(H.nbins):
            binid="{}#{}".format(hname, b)
            ddict[binid]=(H.bins[b].val, H.bins[b].err)

    import json
    with open(sys.argv[2], "w") as f: json.dump(ddict, f, indent=4)
