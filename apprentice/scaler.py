import numpy as np

class Scaler(object):
    def __init__(self, data, a=-1, b=1):
        """
        data   ... np array of points in the original parameter space, file name or dictionary to restore from
        [a,b]  ... target scaling interval

        a, b can be floats or lists.

        Examples:

            D=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[1,4,7],[5,3,9]])

            S=Scaler(D)
            S=Scaler(D, a=0, b=1)
            S=Scaler(D, a=[-1,-2,-1], b=10)
            S=Scaler(D, a=[-1,-2,-1], b=[0,2,4])

        """
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            self._dim  = np.atleast_2d(data).shape[-1]

            if isinstance(a, (int, float)):
                self._a = np.ones(self._dim) * a
            elif isinstance(a, (list, np.ndarray)):
                self._a = np.array(a)
                assert(len(self._a) == self._dim)
            else:
                raise Exception("Data type {} for a not understood in Scaler.__init__".format(type(a)))

            if isinstance(b, (int, float)):
                self._b = np.ones(self._dim) * b
            elif isinstance(b, (list, np.ndarray)):
                self._b = np.array(b)
                assert(len(self._b) == self._dim)
            else:
                raise Exception("Data type {} for b not understood in Scaler.__init__".format(type(b)))

            # Sanity check:
            for i in range(self._dim):
                if self._b[i] <= self._a[i]:
                    raise Exception("Error in defining scale boundaries in coordinate {}: b[{}]={} !> a[{}]={} in Scaler.__init__".format(i,i,b[i],i, a[i]))

            self.mkFromPoints(data)

        else:
            if type(data)==str:
                self.mkFromFile(data)
            elif type(data)==dict:
                self.mkFromDict(data)
            else:
                raise Exception("Data type {} not understood in Scaler.__init__".format(type(data)))

    def mkFromPoints(self, X):
        """
        Main method --- read points X, determine scaling and set attributes.
        """
        self._Xmin = np.amin(X, axis=0)
        self._Xmax = np.amax(X, axis=0)
        self._scaleTerm = (self._b - self._a)/(self._Xmax - self._Xmin)

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
                "a": self._a.tolist(),
                "b": self._b.tolist(),
                "Xmin": self._Xmin.tolist(),
                "scaleTerm":self._scaleTerm.tolist(),
                }

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

    def scale(self, x):
        """
        Scale the point x from the observed range _Xmin, _Xmax to the interval _interval
        (newmax-newmin)/(oldmax-oldmin)*(x-oldmin)+newmin
        """
        return self._scaleTerm*(x - self._Xmin) + self._a

    def unscale(self, x):
        """
        Convert a point from the scaled world back to the real world.
        """
        return self._Xmin + (x-self._a)/self._scaleTerm

    def __str__(self):
        s="Scaler --- translating {}-dimensional points into {}x{}".format(self._dim, self._a, self._b)
        s+="\nOriginal parameter bounds:"
        for a,b in zip(self._Xmin, self._Xmax): s+="\n\t[%f ... %f]"%(a,b)
        return s

    @property
    def center(self):
        return self._ab[:,0] + 0.5*(self._ab[:,1] - self._ab[:,0])

    @property
    def dim(self): return self._dim

    @property
    def sbox(self): return self._ab

    @property
    def box(self):
        """
        The real world parameter box
        """
        _b = np.empty((self.dim, 2))
        _b[:,0]= self(self._ab[:,0])
        _b[:,1]= self(self._ab[:,1])
        return _b




if __name__== "__main__":
    D=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[1,4,7],[5,3,9]])
    S=Scaler(D)
    S.save("testsavescaler.json")
    SS = Scaler("testsavescaler.json")
    assert(all([a==b for a,b in zip(S.scale([3,2,1]), SS.scale([3,2,1]))]))
    print(S)
    print("Scaling:")
    print([1,2,3], "-->", S.scale([1,2,3]))
    print([3,9,1], "-->", S.scale([3,9,1]))
    print("Unscaling:")
    print([-1,-1,-1], "-->", S.unscale([-1,-1,-1]))
    print([1,1,1], "-->", S.unscale([1,1,1]))
    print([0,0,0], "-->", S.unscale([0,0,0]))

    # Now testing with mixed a, b
    S=Scaler(D, a=[-2,-3,-5], b=[10,0,1])
    S.save("testsavescaler.json")
    SS = Scaler("testsavescaler.json")
    print(S)
    print("Scaling:")
    print([1,2,3], "-->", S.scale([1,2,3]))
    print([3,9,1], "-->", S.scale([3,9,1]))
    print("Unscaling:")
    print([10,0,1], "-->", S.unscale([10,0,1]))
    print([-2,-3,-5], "-->", S.unscale([-2,-3,-5]))
    print([0,0,0], "-->", S.unscale([0,0,0]))
