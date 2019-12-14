# OpenCV_sGLOH
This project's goal is to improve the SIFT-Based descriptors' stability to rotations.

# Compilation
To compile the program, please make sure OpenCV is installed on your machine.
To install OpenCV Please refer to it's documentation:
https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html

The build process is handled by CMake, please install CMake as well.
To install CMake please follow this guide:
https://cmake.org/install/

To compile the code on Unix-based system:
    cd OpenCV_sGLOH
    cmake .
    make

To run the test program:
    cd ./build
    ./OpenCV_sGLOH [argument]

The arguments are:
    -k -> Keypoint tests
    -o -> Tests on Oxford Image database
    -m -> Tests on manually taken photos

The results will be saved to the build folder.


# Directory Structure
./build -> All the compiled binaries.  
./src -> The source files (.cc, .cpp).  
./include -> All the header files (.h, .hpp).  
./doc -> All the documentation files.  
./test -> All the test images used.  
./results -> The test results. The subdirectories are as follows:   
    image set name -> keypoint extractor type -> filenames include descriptor type
