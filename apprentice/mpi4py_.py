import numpy as np
class COMM_():
    def __init__(self):
        self.rank = 0
        self.size = 1

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def barrier(self):
        pass

    def gather(self,r):
        return [np.array(r)]

    def scatter(self,s,root=0):
        return s[0]

    def bcast(self, s,root=0):
        return s

class MPI_():
    try:
        from mpi4py import MPI
        COMM_WORLD = MPI.COMM_WORLD
    except ImportError:
        COMM_WORLD = COMM_()

    @staticmethod
    def mpi4py_installed():
        try:
            from mpi4py import MPI
            return True
        except ImportError:
            return False

    @staticmethod
    def print_MPI_message():
        if not MPI_.mpi4py_installed():
            print("WARNING: mpi4py not available. Only 1 rank will be used")