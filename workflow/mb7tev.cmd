


# settings of Pythia 8 wrapper program
Main:numberOfEvents = 100000          ! number of events to generate
Next:numberShowEvent = 0           ! suppress full listing of first events

# random seed
Random:setSeed = on
Random:seed = 1

# Beam parameter settings.
Beams:idA = 2212                ! first beam, p = 2212, pbar = -2212
Beams:idB = 2212                ! second beam, p = 2212, pbar = -2212
Beams:eCM = 7000               ! CM energy of collision



# Minimum Bias process (as taken from one of pythia8 example)
SoftQCD:nonDiffractive = on               ! minimum bias QCD processes
SoftQCD:singleDiffractive = on
SoftQCD:doubleDiffractive = on

# Process setup: min-bias
# Use this for ordinary min-bias (assuming Rivet analysis
# correctly suppresses the diffractive contributions.)
# SoftQCD:all = on # this for min-bias incl diffraction


# Set cuts
# Use this for hard leading-jets in a certain pT window
PhaseSpace:pTHatMin = 0   # min pT
PhaseSpace:pTHatMax = 7000   # max pT

# Use this for hard leading-jets in a certain mHat window
PhaseSpace:mHatMin = 0   # min mHat
PhaseSpace:mHatMax = 7000   # max mHat


# Makes particles with c*tau > 10 mm stable:
ParticleDecays:limitTau0 = On
ParticleDecays:tau0Max = 10.0


# Tune setup:
Tune:pp = 14
MultipartonInteractions:expPow = {expPow}
MultipartonInteractions:pT0Ref = {pT0Ref}
ColourReconnection:range = {range}
