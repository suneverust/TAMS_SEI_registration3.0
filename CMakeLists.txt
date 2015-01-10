# Author: Bo Sun
# Afflication: TAMS, University of Hamburg
# E-Mail: bosun@informatik.uni-hamburg.de
#         user_mail@QQ.com

# We gather the ESSENTIAL files for correlating two spherical funtions
# in SOFT(3), and write this CMakeLists.txt in order to let CMake to # configure and compile the project. 
 
# We use "FindFFTW3.cmake" to let CMake find the FFTW3_INCLUDE_DIR 
# and FFTW3_LIBRARIES automatically, please make sure 
# the "FindFFTW3.cmake" at the right place.

cmake_minimum_required (VERSION 2.8)

project (tams_sei_registration3.0)

find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})
link_directories(${FFTW3_LIBRARIES})

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${PROJECT_SOURCE_DIR}/lib1/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib1)

add_library(tams_soft_fftw_correlate tams_soft_fftw_correlate.cpp)
target_link_libraries(tams_soft_fftw_correlate ${FFTW3_LIBRARIES} libsoft -lm)

add_executable(tams_sei_registration3.0 tams_sei_registration.cpp)

target_link_libraries(tams_sei_registration3.0 ${FFTW3_LIBRARIES}
                                               ${PCL_LIBRARIES}
                                               libsoft
                                               tams_soft_fftw_correlate -lm)
