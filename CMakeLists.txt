cmake_minimum_required( VERSION 2.8 )

project( OpenCV_sGLOH )

## Check if it's windows, if it is, set the OpenCV Directory.
if( WIN32 )
    set( "OpenCV_DIR" "D:/source/repos/opencv/cmakebuild/install" )
endif()

## Find the opencv library.
find_package( OpenCV REQUIRED )

## Use at least C++11
set (CMAKE_CXX_STANDARD 11)

## Setup the build directory.
include_directories( ${OpenCV_INCLUDE_DIRS} include )
set( EXECUTABLE_OUTPUT_PATH build )
add_executable( OpenCV_sGLOH ./src/main.cpp 
			     ./src/gradient.cpp
			     ./src/kp.cpp
			     ./src/options.cpp
			     ./src/sgloh.cpp )

target_link_libraries( OpenCV_sGLOH ${OpenCV_LIBS} )
