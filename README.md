# civ-cross-lyaf

Project: Constraining IGM enrichment and metallicity with the Lyman-alpha and C IV forest correlation function 

This branch contains scripts that run on three different places:

pod: running on [pod cluster in UCSB CSC](https://csc.cnsi.ucsb.edu/)

igm: running on [igm cluster in ENIGMA](http://enigma.physics.ucsb.edu/)

code: running on own computer

Each folder contains CIV_lya_correlation.py, which is the main script of the project. It enables to:

(1) Create Lyman-alpha and C IV forests with .fits files;

(2) Calculate the cross-correlation between the Lyman-alpha and C IV forest;

(3) For a given grid, interpolate the covariance matrix and save the results;

(4) Calculate the likelihood based on interpolated covariance matrix;

(5) Plot the correlation function of the true and inferred model;

(6) Do an inference test.

Each CIV_lya_correlation.py script needs to run with the civ-cross-lyaf branch in enigma and CIV_forest. The other scripts in the folders are mainly used to help the running of the script, including:

(1) Generate the mock data set;

(2) plot the wanted result in custom way.

For more information, please contact Xin Sheng(xinsheng@ucsb.edu)
