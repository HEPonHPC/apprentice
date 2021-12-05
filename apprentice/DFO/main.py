import argparse
from sampleSet import buildInterpolationPoints
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("-s", dest="SEED", type=int, default=2376762,
                        help="Random seed")
    parser.add_argument("-d", dest="WD", type=str, default=None,
                        help="Working Directory")

    args = parser.parse_args()
    buildInterpolationPoints(args.ALGOPARAMS,[],args.SEED,args.WD+"/main_new_params.json")



