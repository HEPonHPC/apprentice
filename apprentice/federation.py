import numpy as np

class Federation(object):
    """
    Federation of teams of approximations
    """

    def __init__(self, fdim, *args):
        """
        The arguments should all be teams that have value methods or filenames
        that point to them
        """

        self.dim_ = fdim

        self.teams_ = []

        for a in args:
            self.addTeam(a)
        pass

    @property
    def dim(self): return self.dim_

    def addTeam(self, team, teamtype=None):
        if type(team)=="str":
            "Implement method that is able to construct team"
            pass
        assert(team.dim == self.dim)
        self.teams_.append(team)


    @property
    def nteams(self): return len(self.teams_)

    @property
    def nplayersperteam(self):
        return [len(t) for t in self.teams_]

    def val(self, x):
        return np.concatenate([t.val(x) for t in self.teams_])


    def hasGradients(self):
        """
        Check if all teams can eval gradients
        """
        pass

    def hasHessians(self):
        """
        Check if all teams can eval hessians
        """
        pass
