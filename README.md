# QForce Tendon Implementation

This directory contains the C++ implementation of the QForce tendon algorithm for real-time muscle control in MuJoCo simulations, along with a Python script for generating comparison videos.

## Overview

The project consists of two main components:
1. **C++ QForce Tendon Implementation** (`qforce_tendon.cpp`) - Processes HDF5 trajectory data and generates control signals
2. **Video Generation Script** (`generate_video.py`) - Creates comparison videos between reference and achieved trajectories

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **C++ Compiler**: GCC 7+ or Clang 6+ (with C++14 support)
- **Python**: 3.7 or higher
- **CMake**: 3.16 or higher

### Required Libraries

#### C++ Dependencies
- **MuJoCo**: Physics simulation engine
- **OSQP**: Quadratic programming solver
- **HDF5**: Hierarchical Data Format library

#### Python Dependencies
See `requirements.txt` in the parent directory for complete list.

## Installation

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake libhdf5-dev
```

#### macOS
```bash
brew install cmake hdf5
```

#### Windows
- Install Visual Studio with C++ support
- Install CMake from https://cmake.org/
- Install HDF5 from https://www.hdfgroup.org/downloads/

### 2. Install MuJoCo

```bash
# Download MuJoCo (requires license)
wget https://github.com/deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz
tar -xf mujoco-2.3.3-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco-2.3.3 ~/.mujoco/

# Set environment variables
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco-2.3.3/bin' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install OSQP

```bash
# Clone and build OSQP
git clone --recursive https://github.com/osqp/osqp
cd osqp
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build . --config Release
sudo cmake --install .
```

### 4. Build the C++ Project

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### C++ QForce Tendon Processing

The C++ program processes HDF5 trajectory data and generates control signals.

#### Basic Usage
```bash
./qforce_tendon <hdf5_file_path> <output_path>
```

#### Example
```bash
./qforce_tendon ../data/sample.hdf5 ../output/
```

#### Input Format
- **HDF5 File**: Contains trajectory data in the format `/emg2pose/timeseries/`
- **Structure**: Compound dataset with `joint_angles` (float32) and `time` (float64) fields

#### Output Format
- **Binary File**: Contains control vectors in the format `{num_vectors, vector_size, data}`
- **File Name**: Based on input HDF5 filename (e.g., `sample.bin`)

#### Progress Tracking
The program shows real-time progress:
```
Progress: 45.2% (55123/121763)
```

### Video Generation

The Python script creates comparison videos between reference and achieved trajectories.

#### Basic Usage
```bash
python generate_video.py --bin_path <binary_file> --hdf5_path <hdf5_file> --output_path <video_directory>
```

#### Example
```bash
python generate_video.py --bin_path ../output/sample.bin --hdf5_path ../data/sample.hdf5 --output_path ../videos/
```

#### Arguments
- `--bin_path`: Path to binary file created by C++ program
- `--hdf5_path`: Path to HDF5 file containing reference trajectory
- `--output_path`: Directory to save the output video

#### Output
- **Video File**: MP4 format showing side-by-side comparison
- **Left Side**: Reference trajectory from HDF5
- **Right Side**: Achieved trajectory using C++ controls

## File Structure

```
qforce_cpp/
├── qforce_tendon.cpp          # Main C++ implementation
├── generate_video.py          # Video generation script
├── CMakeLists.txt             # CMake configuration
├── README.md                  # This file
└── build/                     # Build directory
```

## Algorithm Overview

### QForce Tendon Algorithm

1. **Trajectory Processing**: Reads joint angles and time from HDF5 compound dataset
2. **Inverse Dynamics**: Computes generalized forces for target positions
3. **Muscle Modeling**: Applies tendon dynamics and muscle properties
4. **Quadratic Programming**: Solves for optimal control signals using OSQP
5. **Control Generation**: Produces actuator control values

### Key Components

- **Joint Angle Mapping**: Maps 20 joint angles to 23-dimensional space
- **Time Normalization**: Converts to relative time coordinates
- **Muscle Dynamics**: Models actuator properties and constraints
- **QP Solver**: Uses OSQP for efficient quadratic programming

## Troubleshooting

### Common Issues

#### HDF5 Reading Errors
- **Issue**: "Could not open dataset /emg2pose/timeseries"
- **Solution**: Verify HDF5 file structure and compound dataset format

#### Memory Allocation Errors
- **Issue**: "std::bad_alloc" during processing
- **Solution**: Check HDF5 data types (float32 vs float64) and array dimensions

#### OSQP Solver Issues
- **Issue**: QP solver fails or produces incorrect results
- **Solution**: Verify matrix conditioning and constraint bounds

#### Video Generation Errors
- **Issue**: Missing trajectory data or dimension mismatches
- **Solution**: Ensure binary file and HDF5 file contain matching trajectory lengths

### Debug Information

The C++ program provides detailed debug output:
- Compound dataset structure
- Data type information
- Array dimensions
- Processing progress

## Performance

- **Processing Speed**: ~1000 trajectory points per second (varies by hardware)
- **Memory Usage**: ~50MB for 100k trajectory points
- **Video Generation**: ~30 seconds for 100k points (25 FPS output)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MuJoCo**: Physics simulation engine
- **OSQP**: Quadratic programming solver
- **HDF5**: Data format library
- **MyoSuite**: Musculoskeletal simulation framework 