cmake_minimum_required(VERSION 3.10)
project(qforce_tendon)

set(CMAKE_CXX_STANDARD 17)

# Find HDF5
find_package(HDF5 REQUIRED)

# Local headers and lib
include_directories(/usr/local/include/mujoco)
include_directories(/usr/local/include/osqp)
link_directories(/usr/local/lib)

add_executable(qforce_tendon qforce_tendon.cpp)

target_link_libraries(qforce_tendon mujoco osqp ${HDF5_LIBRARIES})
