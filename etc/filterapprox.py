import apprentice as app
import time


if __name__=="__main__":
    import optparse, os, sys

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                  help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False,
                  help="Turn off messages")
    op.add_option("-o", "--output", dest="OUTPUT",default="filtered.json",
                  help="Training data output file (Default: %default)")
    opts, args = op.parse_args()

    # These are the observable names we want
    obs = sorted(list(set([o.split("#")[0].split("@")[0] for o in app.tools.readObs(args[1])])))

    t0=time.time()
    import json
    with open(args[0]) as f:
        ALL = json.load(f)


    t1=time.time()
    binids = [i for i in ALL.keys() if not i.startswith("__")]
    hnames = [i.split("#")[0] for i in binids]
    keep   = [num for num, hn in enumerate(hnames) if hn in obs]

    xmin = [ALL["__xmin"][k] for k in keep]
    xmax = [ALL["__xmax"][k] for k in keep]

    OUT = {}
    for k in keep:
        bid = binids[k]
        OUT[bid] = ALL[bid]
    OUT["__xmin"] = xmin
    OUT["__xmax"] = xmax
    t2=time.time()

    with open(opts.OUTPUT, "w") as f:
        json.dump(OUT, f, indent=4)
    t3=time.time()

    print("Output of {}/{} objects ({}/{} groups) written to {}.".format(len(keep), len(binids), len(obs), len(set(hnames)), opts.OUTPUT))
    if opts.DEBUG:
        print("Read took {} seconds".format(t1-t0))
        print("Mangling took {} seconds".format(t2-t1))
        print("Write took {} seconds".format(t3-t2))
