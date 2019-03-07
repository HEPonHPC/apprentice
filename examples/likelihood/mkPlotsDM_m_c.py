#!/usr/bin/env python
def mkSPPostCmd(datafile, ix, iy, lx, ly, no_x=False, no_y=False, plot_lim=None):
    cmd = "python -m superplot.super_command %s --xindex=%i --yindex=%i --show_prof_like=0  --plot_description='Two-dimensional posterior pdf.' --show_posterior_mode=0 --show_posterior_mean=0 --show_posterior_pdf=1 --show_posterior_median=0"%(datafile, ix, iy)
    cmd+= " --kde_pdf=1 --show_best_fit=1"
    cmd+=" --leg_position='no legend'"
    if no_x:
        cmd+= " --xlabel=noxlabel"
    else:
        cmd+= " --xlabel='\strut %s'"%lx
    if no_y:
        cmd+= " --ylabel=noylabel"
    else:
        cmd+= " --ylabel='\strut %s'"%ly
    if plot_lim is not None:
        cmd+= " --plot_limits='%f,%f,%f,%f'"%(plot_lim[0],plot_lim[1], plot_lim[2], plot_lim[3])

    cmd+= " --output_file=plot_%i_%i_posterior_xlabel%i_ylabel%i.pdf"%(ix, iy, 1-int(no_x), 1-int(no_y))
    return cmd

def mkSPPostCmdPL(datafile, ix, iy, lx, ly, no_x=False, no_y=False, plot_lim=None):
    cmd = "python -m superplot.super_command %s --xindex=%i --yindex=%i --show_prof_like=1  --plot_description='Two-dimensional profile likelihood.' --show_posterior_mode=0 --show_posterior_mean=0 --show_posterior_pdf=0 --show_posterior_median=0"%(datafile, ix, iy)
    cmd+= " --kde_pdf=1 --show_best_fit=1"
    cmd+=" --leg_position='no legend'"
    if no_x:
        cmd+= " --xlabel=noxlabel"
    else:
        cmd+= " --xlabel='\strut %s'"%lx
    if no_y:
        cmd+= " --ylabel=noylabel"
    else:
        cmd+= " --ylabel='\strut %s'"%ly
    if plot_lim is not None:
        cmd+= " --plot_limits='%f,%f,%f,%f'"%(plot_lim[0],plot_lim[1], plot_lim[2], plot_lim[3])

    cmd+= " --output_file=plot_%i_%i_profile_xlabel%i_ylabel%i.pdf"%(ix, iy, 1-int(no_x), 1-int(no_y))
    return cmd

def mkSPChiCmd(datafile, ix, lx):
    cmd = "python -m superplot.super_command %s --xindex=%i --yindex=3 --show_prof_like=0 --xlabel='%s' --plot_description='One-dimensional chi-squared plot.' --show_posterior_mode=0 --show_posterior_mean=0 --show_posterior_pdf=0 --show_posterior_median=0"%(datafile, ix, lx)
    cmd+= " --output_file=plot_%i_chi2.pdf"%ix
    return cmd

import os, sys


LDICT=dict(zip(['mdm', 'c1','c4','c7', 'c8', 'halo_rhochi', 'halo_vesc', 'halo_v0', 'halo_k', 'cpi', 'cplus'],
    ["$m_\chi$", '$c_1$', '$c_4$', '$c_7$', '$c_8$', 'halo-$\\rho\chi$', 'halo-vesc', 'halo-$v_0$', 'halo-k', "$c_\pi$", "$c_+$"]))


DATA=sys.argv[1]
if len(sys.argv)>2:
    with open(sys.argv[2]) as f:
        import json
        d=json.load(f)
        PNAMES=[x.encode("utf8") for x  in d[list(d.keys())[0]]["scaler"]["pnames"]]
    LABELS=[LDICT[p] for p in PNAMES]
else:
    with open(DATA) as f:
        c=f.readline().strip().split()
        LABELS = ["PAR %i"%(i+1) for i in xrange(len(c)-2)]



RNG=None


def mkLimits(RNG,xname, yname):
    limits=list(RNG[xname])
    limits.extend(list(RNG[yname]))
    return limits


def worker((cmd)):
    import os
    os.system(cmd)

CMDS=[]
# for numx, x in enumerate(LABELS[0:2]):
    # for numy, y in enumerate(LABELS[0:2]):
for numx, x in enumerate(LABELS):
    for numy, y in enumerate(LABELS):
        if numx==numy:continue
        if RNG is None:
            ploli = None
        # CMDS.append(mkSPPostCmd(DATA, numx+2, numy+2, x, y, plot_lim=ploli))
        CMDS.append(mkSPPostCmdPL(DATA, numx+2, numy+2, x, y, plot_lim=ploli))

from multiprocessing import Pool
p = Pool(6)
p.map(worker, CMDS)
