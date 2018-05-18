Dependency
1: Ceres (please use OpenBLAS to build ceres, otherwise it will be very slow), Eigen, Cudaart, jhead, opencv

2. Provide the necessary path in Makefile under SiftGPU folder.

3. Create a "Out" folder. Create "rgb" and "dendogram" folder under "Out" and put the images in jpg in "rgb" folder.

4. Make the makefile under SiftGPU

5. Please keep the libg2o from Optimizer/Thirdparty to LD_LIBRARY_PATH

6. Edit config.txt in "matlab_codes" to give absolute path to the SFM directory


Please add libraries to "LD_LIBRARY_PATH". Examples are:

export LD_LIBRARY_PATH=/home/suvam/sfm/SFM/Optimizer/Thirdparty/g2o/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/suvam/opencv33/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/suvam/sfm/SFM/Optimizer/Thirdparty/g2o/lib:$LD_LIBRARY_PATH

Run the matlab using the following command

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab

7.run the matlab command "run.m" from matlab_codes

