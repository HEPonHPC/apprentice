#!/usr/bin/env python
import sys, os

import glob
PDF   = glob.glob("*posterior_xlabel1_ylabel1.pdf")
pindx=range(2,5)
ND = len(pindx)
print ND

ylabel=1
xlabel=1
allfiles=[]


for numy in range(ND):
    thisline=[]
    for numx in range(ND):
        if numy==numx:
            thisline.append("plot_%i_profile_xlabel%i.pdf"%(numy+2,ylabel))
        elif numx > numy:
            thisline.append("empty.pdf")
        else:
            pll="plot_%i_%i_profile_xlabel%i_ylabel%i.pdf"%(numx+2,numy+2,xlabel,ylabel)
            thisline.append(pll)
    allfiles.append(thisline)
# allfiles.reverse()


cmd="pdfmerge"
for line in allfiles:
    for f in line:
        cmd+=" %s"%f

cmd+=" -o triangle2.pdf"
import os
os.system(cmd)


import os
os.system(cmd)

cmd="pdfnup triangle2.pdf --nup %ix%i"%(ND, ND)
os.system(cmd)

cmd="pdfcrop triangle2-nup.pdf triangle2-nup.pdf"
os.system(cmd)

sys.exit(1)
