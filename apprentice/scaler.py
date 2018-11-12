import numpy as np

class Scaler(object):
    def __init__(self, data, a=-1, b=1):
        """
        data   ... np array of points in the original parameter space, file name or dictionary to restore from
        [a,b]  ... target scaling interval
        """
        self._a = a
        self._b = b
        if type(data)==str:
            self.mkFromFile(data)
        elif type(data)==dict:
            self.mkFromDict(data)
        else:
            self.mkFromPoints(data)

    def mkFromPoints(self, X):
        """
        Main method --- read points X, determine scaling and set attributes.
        """
        self._dim  = np.atleast_2d(X).shape[-1]
        self._ab   = np.ones((self._dim, 2))*[self._a, self._b]
        self._Xmin = np.amin(X, axis=0)
        self._Xmax = np.amax(X, axis=0)
        self._scaleTerm = (self._ab[:,1] - self._ab[:,0])/(self._Xmax - self._Xmin)
        self._XS = self._scale(X)

    def mkFromDict(self, ScalerDict):
        """
        Restore Scaler from properties stored in dictionary.
        The conversion to numpy arrays happens here.
        NOTE: we store the Scaler representation in safe types.
        """

        self._Xmin      = np.array(ScalerDict["Xmin"])
        self._dim       = len(self._Xmin)
        self._scaleTerm = np.array(ScalerDict["scaleTerm"])
        self._a        = np.array(ScalerDict["a"])
        self._b        = np.array(ScalerDict["b"])
        # Just to be safe, owerwrite a and b as set by the constructor
        self._ab   = np.ones((self._dim, 2))*[self._a, self._b]

    def mkFromFile(self, fname):
        """
        Load Scaler representation from YAML file and call
        mkFromDict.
        """
        import json
        with open(fname, "r") as f:
            self.mkFromDict( json.load(f) )

    @property
    def asDict(self):
        """
        JSON friendly representation as dictionary
        """
        return {
                "a": self._a,
                "b": self._b,
                "Xmin": self._Xmin.tolist(),
                "scaleTerm":self._scaleTerm.tolist(),
                }

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

    def _scale(self, x):
        """
        Scale the point x from the observed range _Xmin, _Xmax to the intervale _interval
        (newmax-newmin)/(oldmax-oldmin)*(x-oldmin)+newmin
        """
        return self._scaleTerm*(x - self._Xmin) + self._ab[:,0]

    @property
    def scaledPoints(self):
        return self._XS

    def __call__(self, x):
        """
        Return a single scaled point.
        """
        if len(x)!=self._dim:
            raise Exception("Dimensions incompatible (should be %i)"%self._dim)
        return self._scale(x)

    def __str__(self):
        s="Scaler --- translating %i-dimensional points into [%i,%i]"%(self._dim, self._a, self._b)
        s+="\nOriginal parameter bounds:"
        for a,b in zip(self._Xmin, self._Xmax): s+="\n\t[%f ... %f]"%(a,b)
        return s

    @property
    def dim(self): return self._dim

if __name__== "__main__":
    D=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[1,4,7],[5,3,9]])
    S=Scaler(D)
    S.save("testsavescaler.json")
    SS = Scaler("testsavescaler.json")
    assert(all([a==b for a,b in zip(S([3,2,1]), SS([3,2,1]))]))
    print(S)
    print([1,2,3], "-->", S([1,2,3]))
    print([3,9,1], "-->", S([3,9,1]))
