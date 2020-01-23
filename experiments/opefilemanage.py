import csv
import apprentice as app
import optparse, os, sys, h5py
import numpy as np
if __name__ == "__main__":
    bounds = {"nNoz": [8,10],
                "SOI": [-11,-7],
                'TNA': [1 ,  1.3],
                'NozzleAngle': [73 ,  83],
                'SR': [-2.4 ,  -1],
                'EGR': [0.35 ,  0.5],
                'Pinj': [1400 ,  1800],
                'Pivc': [2.0 ,  2.3],
                'Tivc': [323 ,  373]
              }
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", dest="OUTPUT", default=None, help="Output folder (default: %default)")
    opts, args = op.parse_args()
    os.makedirs(opts.OUTPUT, exist_ok=True)

    X = []
    Y = [[],[],[],[],[],[]]
    header = []
    with open(args[0], mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        flinedone = False
        for row in csv_reader:
            if not flinedone:
                print(f'Column names are {", ".join(row)}')
                for i, (key, value) in enumerate(row.items()):
                    header.append(key)
                flinedone = True
            arrx = []
            for i, (key, value) in enumerate(row.items()):
                if i < 9:
                    arrx.append(value)
                else:
                    Y[i-9].append(value)
            X.append(arrx)
        #     if line_count == 0:
        #         print(f'Column names are {", ".join(row)}')
        #         line_count += 1
        #     # # print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
        #     # line_count += 1
        # print(f'Processed {line_count} lines.')

    X = np.array(X)
    for i in range(len(Y)):
        outfn = "%s" % (header[i + 9])
        y = np.array(Y[i])
        y = np.atleast_2d(y)
        outfile = os.path.join(opts.OUTPUT, outfn)
        np.savetxt(outfile, np.hstack((X, y.T)), delimiter=",", fmt='%s')

    d = {"Xmin": [bounds[header[i]][0] for i in range(9)],
         "Xmax": [bounds[header[i]][1] for i in range(9)],
         "pnames": [header[i] for i in range(9)]}

    import json

    with open(os.path.join(opts.OUTPUT, 'paraminfo.json'), mode='w') as f:
        json.dump(d,f,indent=4)



