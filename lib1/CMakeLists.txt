# Author: Bo Sun
# Afflication: TAMS, University of Hamburg
# E-Mail: bosun@informatik.uni-hamburg.de

# we gather the essential files for correlating two spherical funtions
# in SOFT(2), and write this CMakeLists.txt in orde to use CMake to 
# configure and compile the project. 
 
# We use "FindFFTW3.cmake" to let CMake find the FFTW3_INCLUDE_DIR 
# and FFTW3_LIBRARIES automatically, please make sure the "FindFFTW3.cmake"
# at the right place.

cmake_minimum_required (VERSION 2.8)

project (libsoft)

find_package (FFTW3 REQUIRED)

include_directories (${FFTW3_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/lib1/include)

link_directories (${FFTW3_LIBRARIES})

add_library (libsoft 	csecond.c 
                        makeweights.c
                        s2_primitive.c
                        utils_so3.c
                        utils_vec_cx.c
                        so3_correlate_fftw.c
                        makeWigner.c
                        wignerTransforms_fftw.c
                        s2_cospmls.c
                        soft_fftw.c
                        s2_legendreTransforms.c
                        s2_semi_memo.c)

target_link_libraries (libsoft ${FFTW3_LIBRARIES} -lm)











