#!/bin/bash
# Please type in the following command to run the model:

# 1. chmod +x run_ml.sh
# 2. ./run_ml.sh 

files=("KMeans.py" "BIRCH.py" "MeanShift.py" "GMM.py" "Fuzzy.py" "DBSCAN.py" )

for file in "${files[@]}"; do
    echo "Running ${file}"
    python "${file}"
    echo
done
