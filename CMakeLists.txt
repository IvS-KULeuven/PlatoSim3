# cmake file to build PlatoSim3
#
# Usage: ($ mkdir build)
#        $ cd build
#        $ cmake ..  (alternatively: cmake -D CMAKE_EXPORT_COMPILE_COMMANDS=1 -D CMAKE_CXX_COMPILER=/usr/bin/clang++ .. on mac OSX)
#                    (               mv compile_commands.json ../.                                                                )
#        ($ make clean)
#        $ make -j 4
#        $ make install
#
# After a successful build you can safely remove all files and directories from build/
#
# Notes:
#   - Prefer 'cmake ..' in the build/ directory above 'cmake' in the
#     parent directory, because the former does an out-of-source build.
#   - To create a `compile_commands.json` file for clangd integration:
#        $ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..

# cmake version shouldn't be too old

cmake_minimum_required(VERSION 3.5)


project(PlatoSim)

# Find out which version of the software is being compiled (through git commit hash)
# so that we can include it in the output HDF5 file.

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND git describe --tag
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(GIT_BRANCH "")
  set(GIT_DESCRIBE "")
endif(EXISTS "${CMAKE_SOURCE_DIR}/.git")

# Output some variables during the cmake run for verification

message(STATUS "Git current branch: ${GIT_BRANCH}")
message(STATUS "Git describe: ${GIT_DESCRIBE}")
message(STATUS "CMake source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake binary dir: ${CMAKE_BINARY_DIR}")

# Generate the version.h file that contains the current version of the software
# which will be copied in the HDF5 output file

message(STATUS "Generating version.h")

configure_file(
  ${CMAKE_SOURCE_DIR}/include/version.h.in
  ${CMAKE_BINARY_DIR}/generated/version.h
)

# Specify the locations of the header files

include_directories(
	${CMAKE_BINARY_DIR}/generated
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/hdf5-1.14.6/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/armadillo-6.500.4/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/yaml-cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/fftw-3.3.4/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/Faddeeva/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/zeromq-4.1.4/include
)


# Specify the simulator source files to be compiled, but exclude main.cpp.

file(GLOB platosimSourceFiles ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
list(REMOVE_ITEM platosimSourceFiles ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp)


# Define the main.cpp of platosim separately.

set(platosimMain ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp)


# Define the connectionTest.cpp separately.

set(connectionTestMain ${CMAKE_CURRENT_SOURCE_DIR}/tests/closedLoopTest/connectionTest.cpp)


# Add a platosim library. This way we need to compile the platosim source files
# only once, when using them for platosim.

add_library(platosimLibrary ${platosimSourceFiles})


# Specify the location of the dependency libraries

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/hdf5-1.14.6/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/armadillo-6.500.4/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/yaml-cpp/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/fftw-3.3.4/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/Faddeeva/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/Installs/zeromq-4.1.4/lib
)


# Enable @rpath in the install name for any shared library being built
# Note: in a future version of CMake this will be the default

set(CMAKE_MACOSX_RPATH 1)

# Always use full RPATH, i.e. don't skip the full RPATH for the build tree
# See also: http://www.cmake.org/Wiki/CMake_RPATH_handling
#           http://www.kitware.com/blog/home/post/510

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# When building, don't use the install RPATH already, only later on when installing

SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find the RPATH to be used when installing, but only if it's not a system directory

LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")


# Set the compiler flags
# First those common to both gcc and clang:
# -O3       : level 3 optimization;
# -Wall     : enable all compiler's warning messages

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-deprecated-register -Wno-unused-variable -Wno-return-type")

# Then those specific for the compiler for using C++14 support.

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++14")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++14")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()


# Create the platosim and connectionTest executables

add_executable(platosim ${platosimMain})
add_executable(connectionTest ${connectionTestMain})


# Link the platosim executable with the required libraries

target_link_libraries(platosim platosimLibrary hdf5 hdf5_cpp armadillo yaml-cpp fftw3f fftw3f_threads pthread Faddeeva zmq)
target_link_libraries(connectionTest zmq pthread)

