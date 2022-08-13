cmake_minimum_required(VERSION 3.1.0)
project( webcam )
find_package( OpenCV 4 REQUIRED )
find_package(realsense2 REQUIRED)
find_package(dlib REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable( webcam webcam.cpp )
target_link_libraries( webcam ${OpenCV_LIBS} ${realsense2_LIBRARY} dlib::dlib)
