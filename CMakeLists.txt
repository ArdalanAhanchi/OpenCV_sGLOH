cmake_minimum_required( VERSION 2.8 )

project( OpenCV_ASV_SIFT )

## Check if it's windows, if it is, set the OpenCV Directory.
if( WIN32 ) 
    set("OpenCV_DIR" "D:/opencv")
endif()

## Find the opencv library.
find_package( OpenCV REQUIRED )

## Setup the build directory.
include_directories( ${OpenCV_INCLUDE_DIRS} include )
set( EXECUTABLE_OUTPUT_PATH build )
add_executable( OpenCV_ASV_SIFT ./src/main.cpp )

target_link_libraries( OpenCV_ASV_SIFT ${OpenCV_LIBS} )
