import apprentice

if __name__ == "__main__":
    import sys
    D = apprentice.tools.readH5(sys.argv[1], []) # This reads all bins
    # D = apprentice.readH5(sys.argv[1], [0,1]) # This reads only the first 2 bins

    R = [apprentice.RationalApproximation(*d) for d in D]
