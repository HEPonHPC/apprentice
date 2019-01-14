#!/bin/bash

# This runs the approximation
# The input data is stored in an hdf5 file, outputs written are
# two json files, one with the approximations and another one
# required to scale parameter values for numerical stabiility and
# convenience: approx_flavour.json and approx_flavour.json.scaler
python3 approx_from_h5.py data/flavour.h5  approx_flavour.json

# Run the chi2 optimiser
# This reads the output files from the approximation step as well
# as experimental data. The program does 1001 minimisations serially
# and stores the parameter points in a json file, approx_flavour.json.minimization
python3 chi2_optimizer.py approx_flavour.json data/exerimental_data.json

# Choose a new box
# This analyes the output file from the minimisation step to decide on a
# new sampling box. Scatter plots of the analysed data with the old and
# new boxes are also produced (newbox.pdf, oldbox.pdf). 
python3 new_box.py approx_flavour.json.minimization approx_flavour.json.scaler newbox.json
