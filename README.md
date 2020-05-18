# Data Setup

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup larsoft v08_42_01 -q e19:prof 
setup dunetpc v08_41_01 -q e19:prof
setup larreco v08_27_02 -q e19:prof 
python getSNBData.py

# Requirments

