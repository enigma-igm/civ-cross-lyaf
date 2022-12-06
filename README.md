# Constraining IGM enrichment and metallicity with the Lyman-alpha and C IV forest correlation function 

## Scripts

This branch contains scripts that run on three different places:

pod: scripts that on [pod cluster in UCSB CSC](https://csc.cnsi.ucsb.edu/)

igm: scripts that on [igm cluster in ENIGMA](http://enigma.physics.ucsb.edu/)

code: scripts that on my own computer

The major differences are the path. Each folder contains CIV_lya_correlation.py, which is the main script of the project. It enables to:

(1) Create Lyman-alpha and C IV forests with .fits files;

(2) Calculate the cross-correlation between the Lyman-alpha and C IV forest;

(3) For a given grid, interpolate the covariance matrix and save the results;

(4) Calculate the likelihood based on interpolated covariance matrix;

(5) Plot the correlation function of the true and inferred model;

(6) Do an inference test.

Each CIV_lya_correlation.py script needs to run with the civ-cross-lyaf branch in enigma and CIV_forest. The other scripts in the folders contain additional function, including:

(1) Generate the mock data set;

(2) Plot the wanted result in custom way.

## Figures

The 'present' folder contains some of the representative figures, including:

(1) Skewers of various quantities for the Lyman-alpha and C IV forest;

(2) Cross-correlation function for different parameters;

(3) Covariance matrix and probability distribution.

Most of the results are unpublished, and will be available after the publication. 

## More info

For more information, please contact Xin Sheng(xinsheng@ucsb.edu)
