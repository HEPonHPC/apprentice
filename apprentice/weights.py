import re

# Copied and adapted from professor2
class PointMatcher(object):
    """\
    System for selecting subsets of bins based on a search range
    syntax extended from Professor weight files:
    Path structure: /path/parts/to/histo[syst_variation]@x
                or: /path/parts/to/histo[syst_variation]#n
                or: /path/parts/to/histo[syst_variation]@nmin:nmax
                or: /path/parts/to/histo[syst_variation]#nmin:nmax

    The weight file syntax is derived from YODA path syntax, and allows selecting
    individual bins or bin-ranges either by physical value (an '@' range) or by bin
    number (a '#' range).

    Blank lines and lines starting with a # symbol will be ignored.

    The bin indices used with the # syntax start at 0, and the end index in a
    range is non-inclusive. In the range form, if xmin/nmin or xmax/nmax is left
    blank, it defaults to the accepting all bins from the start of the histogram,
    or all bins to the end of the histogram respectively.

    TODO:
    * Extend to multi-dimensional ranges i.e. @xmin:xmax,#nymin:nymax,...
    * Allow mixed ranges, e.g. @xmin#nmin?
    """

    def __init__(self, patt):
        import sys
        self.re_patt = re.compile(r"([^#@]+)(@[\d\.:]+|#[\d\.:]+)?")
        self.set_patt(patt)

    def set_patt(self, patt):
        "Find path and index/pos parts of patt and assign them to object attrs"
        self.patt = None
        self.path = None
        self.indextype = None
        self.index = None
        if not patt:
            return
        ## Strip separated comments
        patt = re.sub(r"(^|\s+)#.*", "", patt)
        self.patt = patt.strip()
        match = self.re_patt.match(self.patt)
        if match:
            self.path = re.compile(match.group(1))
            if match.group(2):
                # TODO: handle mixed-type ranges?
                self.indextype, indexstr = match.group(2)[0], match.group(2)[1:]
                if self.indextype:
                    if not ":" in indexstr:
                        self.index = float(indexstr)
                    else:
                        indexstr2 = indexstr.split(":", 1)
                        if not indexstr2[0]: indexstr2[0] = "-inf"
                        if not indexstr2[1]: indexstr2[1] = "inf"
                        self.index = [float(istr) for istr in indexstr2]

    def match_path(self, path):
        return self.path.match(path) is not None

    def search_path(self, path):
        return self.path.search(path) is not None

    def match_pos(self, p, low=None, up=None):
        """Decide if a given point p is in the match range.

        p must be an integer at the moment

        """
        if not self.indextype:
            accept = True
        elif self.indextype == "#":
            if type(self.index) is float:
                accept = (p == int(self.index))
            else:
                accept = (p >= self.index[0] and p < self.index[1])
        elif self.indextype == "@":
            if low is None or up is None:
                raise Exception("upper and lower edges cannot be None")
            if type(self.index) is float:
                accept = (self.index >= low and self.index < up)
            else:
                accept = (up > self.index[0] and low <= self.index[1])
        else:
            raise Exception("indextype {} not implemented".format(self.indextype))
        return accept

    def __repr__(self):
        s = "PointMatcher('%s' %s %s %s)" % (self.patt, self.path, self.indextype, self.index)
        return s



def read_pointmatchers(wfile):
    """
    Read a file of PointMatcher definition lines, and return an ordered dict of
    (matcher, string2) pairs, where string2 is the rest of the line.
    """
    import collections
    rtn = collections.OrderedDict() #< need to preserve order
    with open(wfile) as wf:
        for line in wf:
            ## Strip comments (with care due to valid use of # for bin index specification)
            if " #" in line:
                line = line[:line.index(" #")]
            line = line.strip()
            if not line or line[0] == "#":
                continue
            ## Make a new matcher,remainder pair
            parts = line.split(None, 1)
            if len(parts) == 1:
                parts.append("")
            m = PointMatcher(parts[0])
            rtn[m] = parts[1]
    return rtn

def read_pointmatchers_fromlist(wlist):
    """
    Read a file of PointMatcher definition lines, and return an ordered dict of
    (matcher, string2) pairs, where string2 is the rest of the line.
    """
    import collections
    rtn = collections.OrderedDict() #< need to preserve order
    for line in wlist:
        ## Strip comments (with care due to valid use of # for bin index specification)
        if " #" in line:
            line = line[:line.index(" #")]
        line = line.strip()
        if not line or line[0] == "#":
            continue
        ## Make a new matcher,remainder pair
        parts = line.split(None, 1)
        if len(parts) == 1:
            parts.append("")
        m = PointMatcher(parts[0])
        rtn[m] = parts[1]
    return rtn

