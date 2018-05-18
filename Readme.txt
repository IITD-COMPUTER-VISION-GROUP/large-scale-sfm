Dependencies

1: Ceres (please build ceres with OpenBLAS support, for faster computations), Eigen, Cudaart, jhead, opencv

2. Provide the necessary paths in Makefile under SiftGPU folder.

Output Folders:

1. Create a "Out" folder. Create "rgb" and "dendogram" folder under "Out" and put the images in jpg in "rgb" folder.

2. Make the makefile under SiftGPU


Setting Library Paths:

1. Please add the libg2o path from Optimizer/Thirdparty to LD_LIBRARY_PATH

2. Edit config.txt in "matlab_codes" to give absolute path to the lSFM directory


Please add libraries to "LD_LIBRARY_PATH". Examples are:

export LD_LIBRARY_PATH=/home/suvam/sfm/SFM/Optimizer/Thirdparty/g2o/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/suvam/opencv33/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/suvam/sfm/SFM/Optimizer/Thirdparty/g2o/lib:$LD_LIBRARY_PATH

Run the matlab using the following command

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab

Execution:

Run the matlab command "run.m" from matlab_codes

