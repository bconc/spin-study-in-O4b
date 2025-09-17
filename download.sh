#!/bin/bash

OUTDIR="input"
mkdir -p "$OUTDIR"

FILES=(
  "injectionDict_O1O2O3O4a_FAR_1_in_1_semianalytic_SNR_10.pickle"
  "sampleDict_FAR_1_in_1_yr.pickle"
  "sampleDict_O4a_cat_4c4fd2cef_717_date_250401.npy"
)
 
for f in "${FILES[@]}"; do
  wget -P "$OUTDIR" "https://zenodo.org/records/17148537/files/$f"
done

echo "All files downloaded to $OUTDIR/"
