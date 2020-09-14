import apprentice
import time
import unittest
import math
import numpy as np
import h5py

def writeDS(h5file, name, data, compression=4):
    h5file.create_dataset(name=name, data=data, compression=compression)


def readAppSet(fname):
    return apprentice.AppSet(fname)

def mkApprox(fname, set_structures=True, usethese=None):
    import json, apprentice
    with open(fname) as f:
        rd = json.load(f)
    binids = app.tools.sorted_nicely(rd.keys())
    binids = [x for x in binids if not x.startswith("__")]
    if usethese is not None:
        binids = [x for x in binids if x in usethese]

    APP = {}
    for b in binids:
        if "n" in rd[b]: APP[b] = apprentice.RationalApproximation(initDict=rd[b]) # FIXME what about set_structures for rationals?
        else:            APP[b] = apprentice.PolynomialApproximation(initDict=rd[b], set_structures=set_structures)
    return binids, [APP[b] for b in binids]


def readApproxH5(fname, set_structures=True, usethese=None):
    import h5py, apprentice
    with h5py.File(fname, "r") as f:
        # if usethese is None:
        binids = f["id"][:]
        # else:
        pnames = f["pnames"][:]
        a=f["a"][:]
        b=f["b"][:]
        scaleTerm = f["scaleTerm"][:]
        Xmin = f["Xmin"][:]
        Xmax = f["Xmax"][:]
        Vmin = f["Vmin"][:]
        Vmax = f["Vmax"][:]
        dim = f["dim"][:]
        m = f["m"][:]
        n = f["n"][:]
        PC = f["pcoeff"][:]
        QC = f["qcoeff"][:]

    APP = []
    for num, bid in enumerate(binids):
        S={"a":a[num], "b":b[num], "scaleTerm":scaleTerm[num], "Xmin":Xmin[num], "Xmax":Xmax[num], "pnames":pnames[num]}
        vmin = None if np.isnan(Vmin[num]) else Vmin[num]
        vmax = None if np.isnan(Vmax[num]) else Vmax[num]
        if n[num] == 0:
            temp = apprentice.PolynomialApproximation(initDict = {"scaler":S, "pcoeff":PC[num], "m":m[num], "dim":dim[num], "vmin":vmin, "vmax":vmax})
        else:
            temp = apprentice.RationalApproximation(initDict = {"scaler":S, "pcoeff":PC[num], "qcoeff":QC[num], "m":m[num], "n":n[num], "dim":dim[num], "vmin":vmin, "vmax":vmax})
        APP.append(temp)
    return binids, APP

def readApprox(fname):
    import json, apprentice
    with open(fname) as f:
        rd = json.load(f)
    binids = [x for x in rd.keys() if not x.startswith("__")]


    # binids = [x for x in binids if not x.startswith("__")]
    return binids, rd

def convertJson2H5(fin, fout, compression=4):
    a,b = readApprox(fin)
    # edges
    xup = np.array(b["__xmin"])
    xdn = np.array(b["__xmax"])
    M   = [b[bid]["m"]   for bid in a]
    DIM = [b[bid]["dim"] for bid in a]
    # FIXME in future, pure polynomials should just dump n=0
    N = [0 for bid in a]
    for num, bid in enumerate(b):
        try:
            N[num] = b[bid]["n"]
        except:
            pass
    # binids = apprentice.tools.sorted_nicely(rd.keys())

    PC = np.zeros((len(a), apprentice.tools.numCoeffsPoly(max(DIM), max(M))))
    QC = np.zeros((len(a), apprentice.tools.numCoeffsPoly(max(DIM), max(N))))

    # Scaling info
    SC = [b[bid]["scaler"] for bid in a]
    A    = np.zeros((len(a), max(DIM)))
    B    = np.zeros((len(a), max(DIM)))
    ST   = np.zeros((len(a), max(DIM)))
    XMIN = np.zeros((len(a), max(DIM)))
    XMAX = np.zeros((len(a), max(DIM)))
    VMIN = np.full(len(a), np.nan)
    VMAX = np.full(len(a), np.nan)

    # Params
    PN = np.empty((len(a), max(DIM)), dtype=object)

    # for num, bid in enumerate(a):
    for num, (bid, dim) in enumerate(zip(a,DIM)):
        PC[num][:len(b[bid]['pcoeff'])] = b[bid]['pcoeff']
        try:
            QC[num][:len(b[bid]['qcoeff'])] = b[bid]['qcoeff'] # FIXME Polynomials should just have QC =0
        except:
            pass
        A[num][:dim]    = b[bid]["scaler"]["a"]
        B[num][:dim]    = b[bid]["scaler"]["b"]
        ST[num][:dim]   = b[bid]["scaler"]["scaleTerm"]
        XMIN[num][:dim] = b[bid]["scaler"]["Xmin"]
        XMAX[num][:dim] = b[bid]["scaler"]["Xmax"]
        try:
            VMIN[num][:dim] = b[bid]["Vmin"]
            VMAX[num][:dim] = b[bid]["Vmax"]
        except:
            pass
        PN[num][:dim]   = b[bid]['scaler']["pnames"]


    f=h5py.File(fout, mode="w")
    f.create_dataset(name="id",     data=np.array(a,  dtype=bytes), compression=compression)
    f.create_dataset(name="pnames", data=np.array(PN, dtype=bytes), compression=compression)

    f.create_dataset(name="a",      data=A,                         compression=compression)
    f.create_dataset(name="b",      data=B,                         compression=compression)
    f.create_dataset(name="scaleTerm", data=ST,                     compression=compression)
    f.create_dataset(name="Xmin",   data=XMIN,                      compression=compression)
    f.create_dataset(name="Xmax",   data=XMAX,                      compression=compression)
    f.create_dataset(name="xup",   data=xup,                      compression=compression)
    f.create_dataset(name="xdn",   data=xdn,                      compression=compression)
    f.create_dataset(name="Vmin",   data=VMIN,                      compression=compression)
    f.create_dataset(name="Vmax",   data=VMAX,                      compression=compression)
    f.create_dataset(name="dim",    data=np.array(DIM, dtype='i4'), compression=compression)

    f.create_dataset(name="m",      data=np.array(M, dtype='i4'),   compression=compression)
    f.create_dataset(name="n",      data=np.array(N, dtype='i4'),   compression=compression)
    f.create_dataset(name="pcoeff", data=PC,                        compression=compression)
    f.create_dataset(name="qcoeff", data=QC,                        compression=compression)
    f.close()



class ConvertTest(unittest.TestCase):
    def test(self):
        convertJson2H5("test-restructure/la_30.json", "test_30.h5")
        # convertJson2H5("test-restructure/sip_31.json", "test_31.h5")
        # convertJson2H5("test-restructure/sip_13.json", "test_13.h5")
        # convertJson2H5("test-restructure/sip_33.json", "test_33.h5")
        # convertJson2H5("a14_31_val_ipopt.json", "a14.h5")

class ReadH5Test(unittest.TestCase):
    def test(self):
        convertJson2H5("test-restructure/la_30.json", "test_30.h5")
        bh5, ah5 = readApproxH5("test_30.h5")
        bjson, ajson = readApprox("test-restructure/la_30.json")
        P = [2,3]
        for b, a in zip(bh5, ah5):
            assert(b.decode() in bjson)

            idx_json = bjson.index(b.decode())

            approx = apprentice.PolynomialApproximation(initDict=ajson[b.decode()])

            assert(np.sum(approx(P) - a(P))==0)

        convertJson2H5("test-restructure/sip_31.json", "test_31.h5")
        bh5, ah5 = readApproxH5("test_31.h5")
        bjson, ajson = readApprox("test-restructure/sip_31.json")
        P = [2,3]
        for b, a in zip(bh5, ah5):
            assert(b.decode() in bjson)

            idx_json = bjson.index(b.decode())

            approx = apprentice.RationalApproximation(initDict=ajson[b.decode()])

            assert(np.sum(approx(P) - a(P))==0)

if __name__ == '__main__':

    unittest.main()
