#!/bin/bash
# Please type in the following command to run the model:

# 1. chmod +x run_dl.sh
# 2. ./run_dl.sh 

# Run for graph clustering
awk 'NR==238 {$0="    graphhh = 0"} 1' sdcn.py > sdcn_tmp.py && mv sdcn_tmp.py sdcn.py
echo "Running sdcn.py with graph clustering..."
python sdcn.py
echo -e "\n"

# Run for hypergraph clustering
awk 'NR==238 {$0="    graphhh = 1"} 1' sdcn.py > sdcn_tmp.py && mv sdcn_tmp.py sdcn.py
echo "Running sdcn.py with hypergraph clustering..."
python sdcn.py
