#!/usr/bin/env python

import json
import numpy as np

def mkPlot(data, f_out, norm=2):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")

    plt.xlabel("$m$")
    plt.ylabel("$n$")
    plt.xlim((min(xi)-0.5,max(xi)+0.5))
    plt.ylim((min(yi)-0.5,max(yi)+0.5))

    plt.savefig(f_out)
    plt.close('all')

def bar(ax, data, xloc, leglab=None):
    col=["m", "c", "g", "b"]
    for num, ds in enumerate(data):
        y1 = ds['E_rt_mean']
        e1 = ds['E_rt_sd']
        y2 = ds['Eprime_rt_mean']
        e2 = ds['Eprime_rt_sd']

        x = xloc -1 + num*0.15
        if leglab is None:
            ax.bar(x, y1, width=0.15, alpha=0.4, color=col[num],bottom=y2,hatch='//')
            ax.bar(x, y2, width=0.15, alpha=0.7, color=col[num])
        else:
            ax.bar(x, y1, width=0.15, alpha=0.4, color=col[num],bottom=y2,hatch='//')
            ax.bar(x, y2, width=0.15, alpha=0.7, color=col[num],label=leglab[num])

        ax.vlines(x, y1, y1+e1)
        ax.vlines(x, y2, y2+e2)

# python errorplot.py results/plots/Jerrordata.json 0 ../../log/error0.pdf








# python errorplot.py results/plots/Jerrordata.json 10-2 ../../log/error10-2.pdf
# python errorplot.py results/plots/Jerrordata.json 10-6 ../../log/error10-6.pdf
if __name__=="__main__":
    import sys
    with open(sys.argv[1]) as f:
        data=json.load(f)

    tol=sys.argv[2]
    fns = sorted(list(data.keys()), key=lambda x: int(x.split("f")[-1]))

    app = ['rapp', 'rapprd', 'rappsip', 'papp']

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from matplotlib.ticker import MaxNLocator
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    if sys.argv[2].endswith("pgf"):
        mpl.use('pgf')
        pgf_with_custom_preamble = {
            "text.usetex": True,    # use inline math for ticks
            "pgf.rcfonts": False,   # don't setup fonts from rc parameters
            "pgf.preamble": [
                "\\usepackage{amsmath}",         # load additional packages
            ]
        }
        mpl.rcParams.update(pgf_with_custom_preamble)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.xlabel("$m$")
    plt.ylabel("$\\quad E_{r,t}\:,\\quad E^\\prime_{r,t}$",fontsize=20)
    # plt.xlim((min(xi)-0.5,max(xi)+0.5))
    # plt.ylim((min(yi)-0.5,max(yi)+0.5))

    legendlabels=["$r_1(x)$", "$r_2(x)$", "$r_3(x)$", "$p(x)$"]

    for num, f in enumerate(fns):
        temp = [data[f][tol][a] for a in app]
        if num == 0:
            bar(ax, temp, num, legendlabels)
        else:
            bar(ax, temp, num)
        # break

    # xlabels=["\\ref{fn:%s}"%fn for fn in fns]
    xlabels = ['A.1.4','A.1.7','A.1.15','A.1.16','A.1.17']
    if(sys.argv[2] == '0'):
        plt.legend(loc='upper left',fontsize=18)
    plt.yscale("log")
    plt.ylim([10**-13,10**9])
    plt.xticks([x - 0.78 for x in range(len(fns))], xlabels,fontsize=20)
    plt.rc('ytick',labelsize=20)
    plt.rc('xtick',labelsize=20)
    plt.tick_params(labelsize=20)


    assert(sys.argv[1]!=sys.argv[2])
    plt.savefig(sys.argv[3],bbox_inches='tight')
    # from IPython import embed
    # embed()
