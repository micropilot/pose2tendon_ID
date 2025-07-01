#!/usr/bin/env python3
"""
Generate video from C++ QForce tendon output
Reads a binary file and generates a video comparison
Supports local files and Google Cloud Storage (GCS)
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

# GCS imports
try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: Google Cloud Storage not available. Install with: pip install google-cloud-storage")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate video from C++ QForce tendon output")
    parser.add_argument('--bin_path', type=str, required=True, 
                       help='Path to the binary file created by C++ code (local or gs://bucket/path)')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to the HDF5 file containing reference trajectory data (local or gs://bucket/path)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for the video file (local or gs://bucket/path)')
    parser.add_argument('--gcs_project', type=str, default=None,
                       help='Google Cloud project ID (if not set in environment)')
    return parser.parse_args()

def is_gcs_path(path):
    """Check if path is a GCS path"""
    return path.startswith('gs://')

def parse_gcs_path(gcs_path):
    """Parse GCS path into bucket and blob name"""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    # Remove gs:// prefix
    path_without_prefix = gcs_path[5:]
    
    # Split into bucket and blob
    if '/' not in path_without_prefix:
        raise ValueError(f"Invalid GCS path format: {gcs_path}")
    
    bucket_name = path_without_prefix.split('/')[0]
    blob_name = '/'.join(path_without_prefix.split('/')[1:])
    
    return bucket_name, blob_name

def download_from_gcs(gcs_path, local_path):
    """Download file from GCS to local path"""
    if not GCS_AVAILABLE:
        raise ImportError("Google Cloud Storage not available")
    
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download file
    print(f"Downloading {gcs_path} to {local_path}")
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(local_path, gcs_path):
    """Upload file from local path to GCS"""
    if not GCS_AVAILABLE:
        raise ImportError("Google Cloud Storage not available")
    
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload file
    print(f"Uploading {local_path} to {gcs_path}")
    blob.upload_from_filename(local_path)
    return gcs_path

def get_local_path(file_path, temp_dir="/tmp"):
    """Get local path for file, downloading from GCS if necessary"""
    if is_gcs_path(file_path):
        # Create a temporary local path
        filename = os.path.basename(file_path.replace('gs://', '').replace('/', '_'))
        local_path = os.path.join(temp_dir, filename)
        return download_from_gcs(file_path, local_path)
    else:
        return file_path

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

def generate_video_from_cpp_output(bin_path, hdf5_path, output_path, gcs_project=None):
    """Generate video from C++ output binary file"""
    
    # Set GCS project if provided
    if gcs_project and GCS_AVAILABLE:
        os.environ['GOOGLE_CLOUD_PROJECT'] = gcs_project
    
    # Get local paths for input files
    print("Processing input files...")
    local_bin_path = get_local_path(bin_path)
    local_hdf5_path = get_local_path(hdf5_path)
    
    # Check if local files exist
    if not os.path.exists(local_bin_path):
        print(f"Error: Binary file {local_bin_path} not found.")
        return False
    
    if not os.path.exists(local_hdf5_path):
        print(f"Error: HDF5 file {local_hdf5_path} not found.")
        return False
    
    # Load reference trajectory from HDF5 file
    print(f"Reading reference trajectory from {local_hdf5_path}")
    traj = read_hdf5_trajectory(local_hdf5_path)
    print(f"Reference trajectory shape: {traj.shape}")
    
    # Load C++ control output from binary file
    cpp_ctrl = read_cpp_binary(local_bin_path)
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
    
    # Determine output path
    if is_gcs_path(output_path):
        # Create temporary local output path
        bin_basename = os.path.splitext(os.path.basename(bin_path.replace('gs://', '').replace('/', '_')))[0]
        local_output_path = f"/tmp/{bin_basename}.mp4"
        final_output_path = output_path
    else:
        # Local output path
        os.makedirs(output_path, exist_ok=True)
        bin_basename = os.path.splitext(os.path.basename(bin_path))[0]
        local_output_path = os.path.join(output_path, f"{bin_basename}.mp4")
        final_output_path = local_output_path
    
    print(f"Writing {len(frames)} frames to {local_output_path}")
    skvideo.io.vwrite(local_output_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
    
    # Upload to GCS if needed
    if is_gcs_path(output_path):
        upload_to_gcs(local_output_path, final_output_path)
        # Clean up local file
        os.remove(local_output_path)
        print(f"Video uploaded to: {final_output_path}")
    else:
        print(f"Video saved to: {final_output_path}")
    
    # Clean up temporary local files if they were downloaded from GCS
    if is_gcs_path(bin_path) and os.path.exists(local_bin_path):
        os.remove(local_bin_path)
    if is_gcs_path(hdf5_path) and os.path.exists(local_hdf5_path):
        os.remove(local_hdf5_path)
    
    print(f"Video shows: Left = Reference trajectory, Right = C++ QForce tendon output")
    return True

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Binary file: {args.bin_path}")
    print(f"HDF5 file: {args.hdf5_path}")
    print(f"Output path: {args.output_path}")
    
    if not GCS_AVAILABLE and (is_gcs_path(args.bin_path) or is_gcs_path(args.hdf5_path) or is_gcs_path(args.output_path)):
        print("Error: Google Cloud Storage is required for GCS paths but not available.")
        print("Install with: pip install google-cloud-storage")
        sys.exit(1)
    
    success = generate_video_from_cpp_output(args.bin_path, args.hdf5_path, args.output_path, args.gcs_project)
    
    if success:
        print("Video generation completed successfully!")
    else:
        print("Video generation failed!")
        sys.exit(1) 