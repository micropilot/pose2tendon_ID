#!/usr/bin/env python3
"""
Generate video from C++ QForce tendon output
Reads a binary file and generates a video comparison
"""

import os
import sys
import struct
import argparse
import numpy as np
import skvideo.io
import mujoco
import h5py
from tqdm import tqdm
from myosuite.simhive.myo_sim.test_sims import TestSims as loader

def parse_args():
    parser = argparse.ArgumentParser(description="Generate video from C++ QForce tendon output")
    parser.add_argument('--bin_path', type=str, required=True, 
                       help='Path to the binary file created by C++ code')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to the HDF5 file containing reference trajectory data')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for the video file')
    return parser.parse_args()

def read_cpp_binary(filename):
    """Read binary file created by C++ code"""
    with open(filename, 'rb') as f:
        # Read header
        num_vectors = struct.unpack('Q', f.read(8))[0]  # size_t = 8 bytes
        vector_size = struct.unpack('Q', f.read(8))[0]  # size_t = 8 bytes
        
        print(f"Reading {num_vectors} vectors of size {vector_size}")
        
        # Read data
        data = []
        for i in range(num_vectors):
            vector = struct.unpack(f'{vector_size}d', f.read(vector_size * 8))  # double = 8 bytes
            data.append(vector)
    
    return np.array(data)

def read_hdf5_trajectory(hdf5_path):
    """Read trajectory data from HDF5 file (same as C++ code)"""
    with h5py.File(hdf5_path, 'r') as file:
        # Read joint angles and time (same path as C++ code)
        joint_angles = file['emg2pose']['timeseries']['joint_angles'][:]
        time = file['emg2pose']['timeseries']['time'][:]
        
        # Joint angle mapping (same as C++ code)
        map_indexes = {
            3: 1, 4: 0, 5: 2, 6: 3, 7: 5, 8: 4, 9: 6, 10: 7, 11: 9,
            12: 8, 13: 10, 14: 11, 15: 13, 16: 12, 17: 14, 18: 15,
            19: 17, 20: 16, 21: 18, 22: 19
        }
        
        # Preprocess joint angles (same as C++ code)
        njoint_angles = np.zeros((joint_angles.shape[0], 23))
        for k, v in map_indexes.items():
            njoint_angles[:, k] = joint_angles[:, v]
        
        # Convert to relative time (same as C++ code)
        time = time - time.min()
        
        # Convert to trajectory format (same as C++ code)
        traj = np.column_stack([time, njoint_angles])
        
        return traj

def generate_video_from_cpp_output(bin_path, hdf5_path, output_path):
    """Generate video from C++ output binary file"""
    
    # Check if binary file exists
    if not os.path.exists(bin_path):
        print(f"Error: Binary file {bin_path} not found.")
        return False
    
    # Check if HDF5 file exists
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file {hdf5_path} not found.")
        return False
    
    # Load reference trajectory from HDF5 file
    print(f"Reading reference trajectory from {hdf5_path}")
    traj = read_hdf5_trajectory(hdf5_path)
    print(f"Reference trajectory shape: {traj.shape}")
    
    # Load C++ control output from binary file
    cpp_ctrl = read_cpp_binary(bin_path)
    print(f"C++ control data shape: {cpp_ctrl.shape}")
    
    # Ensure we have the same number of controls as trajectory points
    if cpp_ctrl.shape[0] != traj.shape[0]:
        print(f"Warning: Trajectory has {traj.shape[0]} points but C++ output has {cpp_ctrl.shape[0]} controls")
        # Use the minimum of both
        min_points = min(traj.shape[0], cpp_ctrl.shape[0])
        traj = traj[:min_points]
        cpp_ctrl = cpp_ctrl[:min_points]
        print(f"Using first {min_points} points for video generation")
    
    # Initialize models and data
    tausmooth = 5
    model_ref = loader.get_sim(None, 'hand/myohand.xml')
    model_ref.actuator_dynprm[:,2] = tausmooth
    data_ref = mujoco.MjData(model_ref)  # data for reference trajectory
    
    model_test = loader.get_sim(None, 'hand/myohand.xml')
    model_test.actuator_dynprm[:,2] = tausmooth
    data_test = mujoco.MjData(model_test)  # test data for achieved trajectory
    
    # Camera settings
    camera = mujoco.MjvCamera()
    camera.azimuth = 166.553
    camera.distance = 1.178
    camera.elevation = -36.793
    camera.lookat = np.array([-0.93762553, -0.34088276, 0.85067529])
    
    options_ref = mujoco.MjvOption()
    options_ref.flags[:] = 0
    options_ref.geomgroup[1:] = 0
    
    options_test = mujoco.MjvOption()
    options_test.flags[:] = 0
    options_test.flags[4] = 1  # actuator ON
    options_test.geomgroup[1:] = 0
    
    renderer_ref = mujoco.Renderer(model_ref)
    renderer_ref.scene.flags[:] = 0
    
    renderer_test = mujoco.Renderer(model_test)
    renderer_test.scene.flags[:] = 0
    
    # Generation loop
    frames = []
    print("Generating video frames...")
    
    for idx in tqdm(range(traj.shape[0])):
        # Reference trajectory
        data_ref.qpos = traj[idx, 1:]  # Skip timestamp at index 0
        mujoco.mj_step1(model_ref, data_ref)
        
        # Achieved trajectory using C++ controls
        data_test.ctrl = cpp_ctrl[idx]
        mujoco.mj_step(model_test, data_test)
        
        # Generate frames
        if not idx % round(0.3/(model_test.opt.timestep*25)):
            renderer_ref.update_scene(data_ref, camera=camera, scene_option=options_ref)
            frame_ref = renderer_ref.render()
            
            renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
            frame_test = renderer_test.render()
            
            frame_merged = np.append(frame_ref, frame_test, axis=1)
            frames.append(frame_merged)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate output filename based on input binary filename
    bin_basename = os.path.splitext(os.path.basename(bin_path))[0]
    output_name = os.path.join(output_path, f"{bin_basename}.mp4")
    
    print(f"Writing {len(frames)} frames to {output_name}")
    skvideo.io.vwrite(output_name, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
    
    print(f"Video saved to: {output_name}")
    print(f"Video shows: Left = Reference trajectory, Right = C++ QForce tendon output")
    return True

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Binary file: {args.bin_path}")
    print(f"HDF5 file: {args.hdf5_path}")
    print(f"Output path: {args.output_path}")
    
    success = generate_video_from_cpp_output(args.bin_path, args.hdf5_path, args.output_path)
    
    if success:
        print("Video generation completed successfully!")
    else:
        print("Video generation failed!")
        sys.exit(1) 