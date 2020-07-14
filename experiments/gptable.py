import os,sys,json

def getKernelLabel(K):
    kernellablel = {
        'matern32':"M32",
        'matern52':"M52",
        'sqe':"SE",
        'ratquad':"RQ",
        'poly':"PO",
        'or':"OR"
    }
    return kernellablel[K]


cases = ["ne1000_ns100","ne100000_ns1","ne100_ns100","ne10000_ns1"]

metrics = ["meanmsemetric",'sdmsemetric',"chi2metric"]
metricslabel = ["M-MSE", "SD-SE","W-MSE"]


baselogfolder = "../../log"
avgtype = "AvgData"

allbins=[3,4,5]
s = ""
for bno,bin in enumerate(allbins):
    s+="Bin {}\n\n".format(bin)
    header = ""
    for mno, metric in enumerate(metrics):
        s+= "\\textbf{%s}"%metricslabel[mno]
        for cno, case in enumerate(cases):
            PredDataDir = os.path.join(baselogfolder, "SimulationData", "3D_miniapp_PredData",
                                       'gp', avgtype, case)
            file = os.path.join(PredDataDir, "Bin{}_bestmetrics.json".format(bin))
            with open(file, 'r') as f:
                gpds = json.load(f)
            PredDataDir = os.path.join(baselogfolder, "SimulationData", "3D_miniapp_PredData",
                                       'mlhgp', avgtype, case)
            file = os.path.join(PredDataDir, "Bin{}_bestmetrics.json".format(bin))
            with open(file, 'r') as f:
                mlhgpds = json.load(f)

            if gpds['RA']['meanmsemetric'] > mlhgpds['RA']['meanmsemetric']:
                s +="& %.2E "%(gpds['RA'][metric])
            else:
                s += "& %.2E" % (mlhgpds['RA'][metric])
            s+="& %.2E & %.2E " % (gpds['gp'][metric],mlhgpds['mlhgp'][metric])
            if mno==0:
                header+="&\\textbf{RA} & \\textbf{GP}(%s) & \\textbf{HGP}(%s)"%(getKernelLabel(gpds['gp']['bestkernel']),
                            getKernelLabel(mlhgpds['mlhgp']['bestkernel']))


        s+="\\\\\n"
    s += "\n\n\n{}\\\\\\hline \n\n\n".format(header)

print(s)




