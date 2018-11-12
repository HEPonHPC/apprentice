import numpy as np

class Stieltjes(object):
    """
    Calculator for basis orthogonalisation.
    Note that all data is assumed to be scaled.
    """

    def __init__(self, data, max_order=False):
        """
        data      ... np.array or file name to restore from
        max_order ... if given, the ONB will be restricted to polynomials
                      of <= that order, otherwise the maximal order will
                      be determined from the data structure which can be slow
        """
        if type(data)==str:
            self.mkFromFile(data)
        elif type(data)==dict:
            self.mkFromDict(data)
        else:
            from apprentice import tools
            M = tools.maxOrder(*data.shape)
            if max_order and max_order<M: M=max_order
            self._M = M
            self._X = data
            self._dim=data[0].shape[0]
            self._calc()

    @property
    def dim(self):
        """
        Shorthand to get the dimensionality of the parameter space
        """
        return self._dim

    def _reduce(self, M):
        """
        To be called once best orders are determined.
        We drop all unneccessay rows and columns.
        """
        from scipy.special import comb
        Mdof = int(comb(self.dim + M, M))
        self._R = self._R[:Mdof, :Mdof]
        self._Q = self._Q[:,0:Mdof]

    def _calc(self, two_pass=True):
        """
        Stieltjes ONB procedure

        two_pass ... rigger reorthogonalisation
        """

        K = np.atleast_2d(self._X).shape[0]

        from scipy.special import comb
        Mdof = int(comb(self.dim + self._M, self._M))

        Q = np.atleast_2d(np.zeros((K, Mdof)))    # ONB
        R = np.atleast_2d(np.zeros((Mdof, Mdof))) # Recurrence matrix--- stores projections, required to evaluate polynomials

        recInfoInd = np.zeros(Mdof, dtype=np.int)
        recInfoVar = np.zeros(Mdof, dtype=np.int)


        R[0][0] = np.sqrt(K)             # initial (constant vector)
        Q[:,0]  = np.ones((K))/R[0][0]  # insert into matrix

        # print(Q)
        i = 1 # Start algorithm at second vector
        ind = np.zeros((self.dim+1,1)) # Bookkeeping,initially just a bunch of zeros

        # Iterate of degrees
        for m in range(1,self._M + 1):
            # Iterate over variable
            for n in range(self.dim):
                indnn = i
                # Stieltjes/Gram-Schmidt loop.
                #
                # TODO:  This is MGS.  Would CGS be better for
                # parallelization/efficiency?
                for j in range(ind[n], ind[-1]+1):
                    Q[:,i] = self._X[:,n] * Q[:,j]
                    recInfoInd[i] = j;
                    recInfoVar[i] = n;

                    # Orthogonalize. 
                    for k in range(i):
                        R[k][i] = np.dot(Q[:,k],Q[:,i])
                        Q[:,i]  = Q[:,i] - R[k][i]*Q[:,k]
                    R[i][i]=np.linalg.norm(Q[:,i])
                    Q[:,i] = Q[:,i]/R[i][i]

                    # Re-orthogonalize.
                    if two_pass:
                        for k in range(i):
                            Q[:,i] = Q[:,i] - np.dot(Q[:,k], Q[:,i])*Q[:,k]
                        Q[:,i] = Q[:,i]/np.linalg.norm(Q[:,i])
                    i+=1
                ind[n] = indnn
            ind[-1]=i-1

        self._Q = Q
        self._R = R
        self._recInd = recInfoInd
        self._recVar = recInfoVar

    @property
    def R(self): return self._R

    @property
    def Q(self): return self._Q

    def __call__(self, X, dof=None):
        """
        Operator, the arg m is the maximal polynomial order.
        This calls _calc and will set/update Q, R and the recInfo
        """
        if dof is not None:
            return self._recurrence(X, dof)
        else:
            return self._recurrence(X, self._M)


    def _recurrence(self, X, dof):
        """
        Build the recurrence matrix for point X.
        X   ... point of interest
        dof ... degrees of freedom
        """
        import numpy as np
        Q = np.zeros(dof)
        # TODO: make this more efficient
        Q[0] = 1./self._R[0][0]
        for i in range(1, dof):
            Q[i]  = X[self._recVar[i]] * Q[self._recInd[i]]

            for j in range(i): # Note: this was the trouble maker when translating from matlab
                # print i, j, Q[j], self._R[j][i]
                Q[i] -= Q[j]*self._R[j][i]

            Q[i] /= self._R[i][i]

        return Q

    @property
    def asDict(self):
        return {
                "Q" : self.Q.tolist(),
                "R" : self.R.tolist(),
                "dim" : self._dim,
                "M" : self._M,
                "recInd": self._recInd.tolist(),
                "recVar": self._recVar.tolist(),
                }

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump( self.asDict, f)

    def mkFromFile(self, fname):
        import json
        with open(fname, "r") as f:
            self.mkFromDict( json.load(f) )

    def mkFromDict(self, ONBDict):
        self._Q = np.array(ONBDict["Q"])
        self._R = np.array(ONBDict["R"])
        self._dim = int(ONBDict["dim"])
        self._M = int(ONBDict["M"])
        self._recInd = np.array(ONBDict["recInd"])
        self._recVar = np.array(ONBDict["recVar"])

    def __str__(self):
        s="Stieltjes ONB generator"
        return s



if __name__== "__main__":
    D=np.array([[1.,2.], [3.,4.], [5.,6.], [7.,8.], [9.,1], [4,7], [5,3] ])
    O=Stieltjes(D)

    from apprentice import scaler
    S=scaler.Scaler(D)

    print(O(D[0]))
    print(O(S(D[0])))
    O.save("testSaveONB.json")

    O2 = Stieltjes("testSaveONB.json")
    print(O2(D[0]))
    print(O2(S(D[0])))
