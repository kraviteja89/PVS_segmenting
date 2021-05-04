# PVS_segmenting
Matlab and Python code for segmenting of paravascular spaces from fast 3D scanning with two-photon microscopy using CAG-EGFP (https://www.jax.org/strain/003291) mice

Copyright Ravi Kedarasetti, Postdoctoral researcher, Pennsylvania State University, PA, USA

Written for MATLAB 2019a

and Python 3.8 with the following packages
numpy 1.19.5
Keras 2.4.3
tensorflow 2.4.1
matplotlib 3.3.4
keras-vgg-face 0.6
h5py 3.2.1
Pillow 8.1.2 


To run the analysis, Have the green and red channel image stacks( in .TIF format) in subfolders named "Green" and "Red" in the same parent directory
The filename of the red and green channel images should be the same.
The parent directory should also contain analog accelerometer data (in binary text file) for the spherical treadmill, with the same filename as the red and green channel data


First run prep_for_python.m 
You will be prompted to select a 224x224 mask and create the test segmenting data for all the selected slices

Run PVS_one_off.py
to train the neural network and perform the segmenting

Run python_post_process.m to calculate vessel diameters and make 3D video.

Update animal name and file name for all three files
Sample result: https://sites.psu.edu/raviteja/research/pvs-segmentation-with-cnn/

The current method uses transfer learning, where the first 4 convolutional layers of vgg-face16 are used.
