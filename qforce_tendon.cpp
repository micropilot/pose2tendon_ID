#include "mujoco.h"
#include "stdio.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <osqp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <map>

std::vector<double> solve_qp(const std::vector<std::vector<double>>& P_dense_input,
                             const std::vector<double>& q,
                             const std::vector<double>& lb,
                             const std::vector<double>& ub,
                             const std::vector<double>* x0_ptr = nullptr) {
    int n = q.size();

    // Copy and symmetrize P_dense
    std::vector<std::vector<double>> P_dense = P_dense_input;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double avg = 0.5 * (P_dense[i][j] + P_dense[j][i]);
            P_dense[i][j] = avg;
            P_dense[j][i] = avg;
        }
    }
        
    // Count non-zeros in upper triangular part
    int nnz = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (P_dense[i][j] != 0.0) nnz++;
        }
    }

    // Create CSC for P (upper triangular only)
    OSQPFloat* P_x = new OSQPFloat[nnz];
    OSQPInt* P_i = new OSQPInt[nnz];
    OSQPInt* P_p = new OSQPInt[n + 1];

    int idx = 0;
    P_p[0] = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (P_dense[i][j] != 0.0) {
                P_x[idx] = P_dense[i][j];
                P_i[idx] = i;
                idx++;
            }
        }
        P_p[j + 1] = idx;
    }
    OSQPCscMatrix* P = OSQPCscMatrix_new(n, n, nnz, P_x, P_i, P_p);

    // Build A as identity
    OSQPFloat* A_x = new OSQPFloat[n];
    OSQPInt* A_i = new OSQPInt[n];
    OSQPInt* A_p = new OSQPInt[n + 1];
    for (int j = 0; j < n; j++) {
        A_x[j] = 1.0;
        A_i[j] = j;
        A_p[j] = j;
    }
    A_p[n] = n;
    OSQPCscMatrix* A = OSQPCscMatrix_new(n, n, n, A_x, A_i, A_p);

    // Create settings
    OSQPSettings* settings = OSQPSettings_new();
    settings->verbose = false;
    settings->rho = 0.1;
    settings->alpha = 1.6;
    settings->eps_abs = 1e-3;
    settings->eps_rel = 1e-3;
    settings->polishing = false;

    // Setup solver
    OSQPSolver* solver = nullptr;
    int exitflag = osqp_setup(&solver, P, const_cast<OSQPFloat*>(q.data()), A,
                              const_cast<OSQPFloat*>(lb.data()), const_cast<OSQPFloat*>(ub.data()), n, n, settings);


    if (!exitflag) exitflag = osqp_solve(solver);

    // Extract solution
    std::vector<double> result(n, 0.0);
    if (!exitflag && solver->solution && solver->solution->x) {
        for (int i = 0; i < n; i++) {
            result[i] = solver->solution->x[i];
        }
    }

    // Cleanup
    osqp_cleanup(solver);
    OSQPCscMatrix_free(A);
    OSQPCscMatrix_free(P);
    OSQPSettings_free(settings);
    delete[] A_x;
    delete[] A_i;
    delete[] A_p;

    return result;
}

int main(int argc, const char** argv) {
    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <hdf5_file_path> <output_path>\n", argv[0]);
        printf("Example: %s data/emg2pose_data/sample.hdf5 output/\n", argv[0]);
        return 1;
    }
    
    const char* hdf5_file_path = argv[1];
    const char* output_path = argv[2];
    std::string file_path_str(hdf5_file_path);
    std::vector<std::vector<double>> trajectory_data;
    
    // Load HDF5 file
    hid_t file_id = H5Fopen(hdf5_file_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Could not open HDF5 file: %s\n", hdf5_file_path);
        return 1;
    }
    
    // Read the compound dataset - same path as Python: file['emg2pose']['timeseries']
    hid_t dataset_id = H5Dopen2(file_id, "/emg2pose/timeseries", H5P_DEFAULT);
    if (dataset_id < 0) {
        printf("Could not open dataset /emg2pose/timeseries\n");
        H5Fclose(file_id);
        return 1;
    }
    
    // Get dataset type and check if it's compound
    hid_t dataset_type = H5Dget_type(dataset_id);
    H5T_class_t type_class = H5Tget_class(dataset_type);
    
    if (type_class != H5T_COMPOUND) {
        printf("Dataset is not a compound type. Type class: %d\n", type_class);
        H5Tclose(dataset_type);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return 1;
    }
    
    // Get compound type info
    size_t compound_size = H5Tget_size(dataset_type);
    int num_members = H5Tget_nmembers(dataset_type);
    printf("Compound dataset: size=%zu, members=%d\n", compound_size, num_members);
    
    // Print member information
    for (int i = 0; i < num_members; i++) {
        char* member_name = H5Tget_member_name(dataset_type, i);
        size_t member_offset = H5Tget_member_offset(dataset_type, i);
        hid_t member_type = H5Tget_member_type(dataset_type, i);
        size_t member_size = H5Tget_size(member_type);
        H5T_class_t member_class = H5Tget_class(member_type);
        
        printf("Member %d: name='%s', offset=%zu, size=%zu, class=%d\n", 
               i, member_name, member_offset, member_size, member_class);
        
        free(member_name);
        H5Tclose(member_type);
    }
    
    // Get dataset dimensions (should be 1D)
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    
    printf("Dataset shape: %zu (1D)\n", dims[0]);
    
    // Read the compound data
    std::vector<char> raw_data(dims[0] * compound_size);
    H5Dread(dataset_id, dataset_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_data.data());
    
    // Find the joint_angles and time members
    int joint_angles_idx = -1;
    int time_idx = -1;
    size_t joint_angles_offset = 0;
    size_t time_offset = 0;
    size_t joint_angles_size = 0;
    size_t time_size = 0;
    
    for (int i = 0; i < num_members; i++) {
        char* member_name = H5Tget_member_name(dataset_type, i);
        if (strcmp(member_name, "joint_angles") == 0) {
            joint_angles_idx = i;
            joint_angles_offset = H5Tget_member_offset(dataset_type, i);
            hid_t member_type = H5Tget_member_type(dataset_type, i);
            joint_angles_size = H5Tget_size(member_type);
            H5Tclose(member_type);
        } else if (strcmp(member_name, "time") == 0) {
            time_idx = i;
            time_offset = H5Tget_member_offset(dataset_type, i);
            hid_t member_type = H5Tget_member_type(dataset_type, i);
            time_size = H5Tget_size(member_type);
            H5Tclose(member_type);
        }
        free(member_name);
    }
    
    if (joint_angles_idx == -1 || time_idx == -1) {
        printf("Could not find joint_angles or time members in compound dataset\n");
        H5Tclose(dataset_type);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return 1;
    }
    
    printf("Found joint_angles at offset %zu (size %zu), time at offset %zu (size %zu)\n", 
           joint_angles_offset, joint_angles_size, time_offset, time_size);
    
    // Extract joint angles and time data
    std::vector<float> joint_angles_data;  // Changed to float for float32
    std::vector<double> time_data;         // Keep as double for float64
    
    // Get the actual member types to determine sizes correctly
    hid_t joint_angles_member_type = H5Tget_member_type(dataset_type, joint_angles_idx);
    hid_t time_member_type = H5Tget_member_type(dataset_type, time_idx);
    
    size_t joint_angles_element_size = H5Tget_size(joint_angles_member_type);
    size_t time_element_size = H5Tget_size(time_member_type);
    
    printf("joint_angles element size: %zu bytes, time element size: %zu bytes\n", 
           joint_angles_element_size, time_element_size);
    
    // Check if joint_angles is an array or single value
    H5T_class_t joint_angles_class = H5Tget_class(joint_angles_member_type);
    H5T_class_t time_class = H5Tget_class(time_member_type);
    
    printf("joint_angles class: %d, time class: %d\n", joint_angles_class, time_class);
    
    // Check if joint_angles is an array type
    size_t num_joint_angles_per_element = 1;  // Default to single value
    
    if (joint_angles_class == H5T_ARRAY) {
        int array_rank = H5Tget_array_ndims(joint_angles_member_type);
        hsize_t array_dims[10];
        H5Tget_array_dims(joint_angles_member_type, array_dims);
        
        printf("joint_angles is an array with rank %d, dimensions: ", array_rank);
        for (int d = 0; d < array_rank; d++) {
            printf("%zu ", array_dims[d]);
        }
        printf("\n");
        
        // Get the base type of the array
        hid_t base_type = H5Tget_super(joint_angles_member_type);
        H5T_class_t base_class = H5Tget_class(base_type);
        size_t base_size = H5Tget_size(base_type);
        printf("joint_angles base type: class=%d, size=%zu\n", base_class, base_size);
        H5Tclose(base_type);
        
        // Calculate total elements in the array
        size_t total_elements = 1;
        for (int d = 0; d < array_rank; d++) {
            total_elements *= array_dims[d];
        }
        num_joint_angles_per_element = total_elements;
    } else {
        printf("joint_angles is not an array, treating as single value\n");
    }
    
    for (size_t i = 0; i < dims[0]; i++) {
        char* element_ptr = raw_data.data() + i * compound_size;
        
        // Extract time (float64)
        double time_val = *reinterpret_cast<double*>(element_ptr + time_offset);
        time_data.push_back(time_val);
        
        // Extract joint angles (float32)
        float* joint_angles_ptr = reinterpret_cast<float*>(element_ptr + joint_angles_offset);
        for (size_t j = 0; j < num_joint_angles_per_element; j++) {
            joint_angles_data.push_back(joint_angles_ptr[j]);
        }
    }
    
    printf("Extracted %zu time points and %zu joint angle values (%zu per time point)\n", 
           time_data.size(), joint_angles_data.size(), num_joint_angles_per_element);
    
    // Clean up member types
    H5Tclose(joint_angles_member_type);
    H5Tclose(time_member_type);
    
    // Joint angle mapping (same as Python code)
    std::map<int, int> map_indexes = {
        {3, 1}, {4, 0}, {5, 2}, {6, 3}, {7, 5}, {8, 4}, {9, 6}, {10, 7}, {11, 9},
        {12, 8}, {13, 10}, {14, 11}, {15, 13}, {16, 12}, {17, 14}, {18, 15},
        {19, 17}, {20, 16}, {21, 18}, {22, 19}
    };
    
    // Preprocess joint angles (same as Python: njoint_angles = np.zeros((joint_angles.shape[0], 23)))
    size_t num_time_points = time_data.size();
    std::vector<std::vector<double>> njoint_angles(num_time_points, std::vector<double>(23, 0.0));
    
    // Apply mapping (same as Python: njoint_angles[:, k] = joint_angles[:, v])
    for (size_t i = 0; i < num_time_points; i++) {
        for (const auto& mapping : map_indexes) {
            int target_idx = mapping.first;
            int source_idx = mapping.second;
            if (source_idx < num_joint_angles_per_element) {
                njoint_angles[i][target_idx] = joint_angles_data[i * num_joint_angles_per_element + source_idx];
            }
        }
    }
    
    // Convert to relative time (same as Python: time = time - time.min())
    double min_time = time_data[0];
    for (size_t i = 1; i < time_data.size(); i++) {
        if (time_data[i] < min_time) min_time = time_data[i];
    }
    for (size_t i = 0; i < time_data.size(); i++) {
        time_data[i] -= min_time;
    }
    
    // Convert to trajectory format (same as Python: traj = np.column_stack([time, njoint_angles]))
    for (size_t i = 0; i < num_time_points; i++) {
        std::vector<double> row;
        row.push_back(time_data[i]);  // Add timestamp
        for (size_t j = 0; j < 23; j++) {  // Use 23 joint angles as in Python
            row.push_back(njoint_angles[i][j]);
        }
        trajectory_data.push_back(row);
    }
    
    // Cleanup HDF5 resources
    H5Tclose(dataset_type);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    printf("Loaded %zu rows of trajectory data from HDF5\n", trajectory_data.size());

    // Load model
    mjModel* m = mj_loadXML("myo_sim/hand/myohand.xml", NULL, NULL, 0);

    // Set tausmooth parameter (same as Python: model1.actuator_dynprm[:,2] = tausmooth)
    double tausmooth = 5.0;
    for (int i = 0; i < m->nu; i++) {
        m->actuator_dynprm[i*mjNDYN + 2] = tausmooth;  // Set the 3rd parameter (index 2) for each actuator
    }
    printf("Set tausmooth to %f for all %d actuators\n", tausmooth, m->nu);

    // Create data
    mjData* d = mj_makeData(m);
    
    // Function to compute generalized force (equivalent to Python get_qfrc)
    auto get_qfrc = [&](const std::vector<double>& target_qpos) -> std::vector<double> {
        // Create a copy of data for computation
        mjData* d_copy = mj_makeData(m);
        
        // Manually copy relevant data from original to copy
        for (int i = 0; i < m->nq; i++) {
            d_copy->qpos[i] = d->qpos[i];
        }
        for (int i = 0; i < m->nv; i++) {
            d_copy->qvel[i] = d->qvel[i];
        }
        for (int i = 0; i < m->nu; i++) {
            d_copy->ctrl[i] = d->ctrl[i];
        }
        for (int i = 0; i < m->nu; i++) {
            d_copy->act[i] = d->act[i];
        }
        
        // Compute acceleration: qacc = ((target_qpos - qpos) / timestep - qvel) / timestep
        double timestep = m->opt.timestep;
        for (int i = 0; i < m->nv; i++) {
            d_copy->qacc[i] = ((target_qpos[i] - d_copy->qpos[i]) / timestep - d_copy->qvel[i]) / timestep;
        }
        
        // Compute inverse dynamics
        mj_inverse(m, d_copy);
        
        // Extract qfrc_inverse
        std::vector<double> qfrc(m->nv);
        for (int i = 0; i < m->nv; i++) {
            qfrc[i] = d_copy->qfrc_inverse[i];
        }
        
        mj_deleteData(d_copy);
        return qfrc;
    };

    // Function to compute control (equivalent to Python get_ctrl)
    auto get_ctrl = [&](const std::vector<double>& target_qpos, 
                       const std::vector<double>& qfrc,
                       double qfrc_scaler = 100.0,
                       double qvel_scaler = 5.0) -> std::vector<double> {
        
        int nv = m->nv, nu = m->nu;
        
        // FIRST: Extract current actuator states from ORIGINAL data
        std::vector<double> act(nu);
        std::vector<double> ctrl0(nu);
        for (int i = 0; i < m->nu; i++) {
            act[i] = d->act[i];  // Use original data
            ctrl0[i] = d->ctrl[i];  // Use original data
        }

        
        // SECOND: Compute actuator dynamics parameters using original actuator states
        double ts = m->opt.timestep;
        std::vector<double> tA(nu), tD(nu);
        std::vector<double> t1(nu), t2(nu);
        
        
        for (int i = 0; i < m->nu; i++) {
            // Try different indices to find correct parameters
            // Since mjNDYN = 10, maybe the parameters are at different positions
            tA[i] = m->actuator_dynprm[i*mjNDYN + 0] * (0.5 + 1.5 * act[i]);  // Try index 3
            tD[i] = m->actuator_dynprm[i*mjNDYN + 1] / (0.5 + 1.5 * act[i]);  // Try index 4
            t1[i] = (tA[i] - tD[i]) * 1.875 / m->actuator_dynprm[i*mjNDYN + 2];  // Try index 5
            t2[i] = (tA[i] + tD[i]) * 0.5;
        }
        
        // THIRD: Create a copy of data for computation
        mjData* d_copy = mj_copyData(NULL, m, d);

        // Set target position and velocity for gain/bias computation
        for (int i = 0; i < m->nv; i++) {
            d_copy->qpos[i] = target_qpos[i];
            d_copy->qvel[i] = ((target_qpos[i] - d->qpos[i]) / ts) / qvel_scaler;
        }
        
        
        // Compute step1 to get actuator properties
        mj_step1(m, d_copy);
        
        // Compute gain and bias for each actuator
        mjtNum *gain = mj_stackAllocNum(d_copy, nu);
        mjtNum *bias = mj_stackAllocNum(d_copy, nu);
        for (int i = 0; i < m->nu; i++) {
            double length = d_copy->actuator_length[i];
            double lengthrange[2] = {m->actuator_lengthrange[i*2], m->actuator_lengthrange[i*2+1]};  // (nu x 2) structure
            double velocity = d_copy->actuator_velocity[i];
            double acc0 = m->actuator_acc0[i];
            
            // Extract bias and gain parameters
            // actuator_biasprm is (nu x mjNBIAS), actuator_gainprm is (nu x mjNGAIN)
            double prmb[9], prmg[9];
            for (int j = 0; j < 9; j++) {
                prmb[j] = m->actuator_biasprm[i * mjNBIAS + j];  // Correct stride
                prmg[j] = m->actuator_gainprm[i * mjNGAIN + j];  // Correct stride
            }
            
            // Compute bias and gain (simplified versions)
            bias[i] = mju_muscleBias(length, lengthrange, acc0, prmb);
            gain[i] = mju_muscleGain(length, velocity, lengthrange, acc0, prmg);
            gain[i] = std::min(-1.0, gain[i]);
        }

        mjtNum *AM_sparse = mj_stackAllocNum(d_copy, nu*nv);
        mju_sparse2dense(AM_sparse, d_copy->actuator_moment, nu, nv, d_copy->moment_rownnz,
                        d_copy->moment_rowadr, d_copy->moment_colind);
        
        // AM is the transpose of AM_sparse, so it has dimensions (nv x nu)
        mjtNum *AM = mj_stackAllocNum(d_copy, nv*nu);
        mju_transpose(AM, AM_sparse, nu, nv);  // Transpose AM_sparse (nu x nv) to AM (nv x nu)
        mjtNum *P = mj_stackAllocNum(d_copy, nu*nu);
        mju_mulMatTMat(P, AM, AM, nv, nu, nu);  // P = AM^T @ AM where AM is (nv x nu), so AM^T is (nu x nv)
        
        // Multiply P by 2 to get final P matrix
        mju_scl(P, P, 2.0, nu*nu);
        
        // k = AM @ (gain * act) + AM @ bias - (qfrc / qfrc_scaler)
        mjtNum *k = mj_stackAllocNum(d_copy, nv);
        mjtNum *gain_act = mj_stackAllocNum(d_copy, nu);
        mjtNum *am_gain_act = mj_stackAllocNum(d_copy, nv);
        mjtNum *am_bias = mj_stackAllocNum(d_copy, nv);
        mjtNum *qfrc_scaled = mj_stackAllocNum(d_copy, nv);
        mjtNum *last_two_terms = mj_stackAllocNum(d_copy, nv);

        // Compute gain * act
        for (int i = 0; i < nu; i++) {
            gain_act[i] = gain[i] * act[i];
        }
        // am_gain_act = AM @ (gain * act)
        mju_mulMatVec(am_gain_act, AM, gain_act, nv, nu);
        // am_bias = AM @ bias
        mju_mulMatVec(am_bias, AM, bias, nv, nu);
        // -qfrc_scaled = (qfrc / qfrc_scaler)
        mju_scl(qfrc_scaled, qfrc.data(), -1.0/qfrc_scaler, nv);
        // last_two_terms = am_bias + qfrc_scaled
        mju_add(last_two_terms, am_bias, qfrc_scaled, nv);
        // k = am_gain_act + last_two_terms
        mju_add(k, am_gain_act, last_two_terms, nv);
        // q = 2 * k @ AM
        mjtNum *q = mj_stackAllocNum(d_copy, nu);
        mju_mulMatTVec(q, AM, k, nv, nu);  // q = AM^T @ k = k @ AM (since k is a vector)
        mju_scl(q, q, 2.0, nu);
        

        // lb = gain * (1 - act) * ts / (t2 + t1 * (1 - act))
        mjtNum *lb = mj_stackAllocNum(d_copy, nu);
        for (int i = 0; i < nu; i++) {
            lb[i] = gain[i] * (1 - act[i]) * ts / (t2[i] + t1[i] * (1 - act[i]));
        }
        
        // ub = -gain * act * ts / (t2 - t1 * act)
        mjtNum *ub = mj_stackAllocNum(d_copy, nu);
        for (int i = 0; i < nu; i++) {
            ub[i] = -gain[i] * act[i] * ts / (t2[i] - t1[i] * act[i]);
        }
        
        // Convert mjtNum arrays to std::vector for solve_qp
        std::vector<std::vector<double>> P_vec(nu, std::vector<double>(nu, 0.0));
        std::vector<double> q_vec(nu, 0.0);
        std::vector<double> lb_vec(nu, 0.0);
        std::vector<double> ub_vec(nu, 0.0);
        
        // Copy P matrix
        for (int i = 0; i < nu; i++) {
            for (int j = 0; j < nu; j++) {
                P_vec[i][j] = P[i * nu + j];
            }
        }
        
        // Copy vectors
        for (int i = 0; i < nu; i++) {
            q_vec[i] = q[i];
            lb_vec[i] = lb[i];
            ub_vec[i] = ub[i];
        }
        
        std::vector<double> x = solve_qp(P_vec, q_vec, lb_vec, ub_vec);
        // Compute control
        std::vector<double> ctrl(m->nu);
        for (int i = 0; i < m->nu; i++) {
            ctrl[i] = act[i] + x[i] * t2[i] / (gain[i] * ts - x[i] * t1[i]);
            ctrl[i] = std::max(0.0, std::min(1.0, ctrl[i]));  // Clip to [0, 1]
        }
        mj_deleteData(d_copy);
        return ctrl;
    };

    // Iterate through trajectory
    printf("Processing trajectory with %zu points...\n", trajectory_data.size());
    
    // Collect all controls
    std::vector<std::vector<double>> all_ctrl;
    
    // Progress tracking
    size_t progress_interval = std::max(1UL, trajectory_data.size() / 100);  // Show progress every 1% or at least every point
    
    for (size_t i = 0; i < trajectory_data.size(); i++) {
        // Show progress
        if (i % progress_interval == 0 || i == trajectory_data.size() - 1) {
            double progress = (double)(i + 1) / trajectory_data.size() * 100.0;
            printf("\rProgress: %.1f%% (%zu/%zu)", progress, i + 1, trajectory_data.size());
            fflush(stdout);
        }
        
        const auto& row = trajectory_data[i];
        
        // Extract target position (skip timestamp at index 0)
        std::vector<double> target_qpos(row.begin() + 1, row.begin() + 1 + m->nv);
        // Compute generalized force
        std::vector<double> qfrc = get_qfrc(target_qpos);
        std::vector<double> ctrl = get_ctrl(target_qpos, qfrc);
        
        // Store control
        all_ctrl.push_back(ctrl);
        
        // Apply control to simulation
        for (int j = 0; j < m->nu; j++) {
            d->ctrl[j] = ctrl[j];
        }
        // Step the simulation
        mj_step(m, d);
        
    }
    
    printf("\n"); // New line after progress bar

    // Generate output file name based on input file name
    std::string input_filename = file_path_str;
    size_t last_slash = input_filename.find_last_of("/\\");
    size_t last_dot = input_filename.find_last_of(".");
    std::string base_name = input_filename.substr(last_slash + 1, last_dot - last_slash - 1);
    std::string output_filename = std::string(output_path) + "/" + base_name + ".bin";
    
    // Save controls to binary file for full precision
    FILE* bin_file = fopen(output_filename.c_str(), "wb");
    if (bin_file) {
        // Write header: number of control vectors and size of each vector
        size_t num_vectors = all_ctrl.size();
        size_t vector_size = m->nu;
        fwrite(&num_vectors, sizeof(size_t), 1, bin_file);
        fwrite(&vector_size, sizeof(size_t), 1, bin_file);
        
        // Write control data as raw binary
        for (const auto& ctrl : all_ctrl) {
            fwrite(ctrl.data(), sizeof(double), ctrl.size(), bin_file);
        }
        
        fclose(bin_file);
        printf("Saved %zu control vectors to %s\n", all_ctrl.size(), output_filename.c_str());
    } else {
        printf("Failed to open %s for writing\n", output_filename.c_str());
    }

    // Cleanup
    mj_deleteData(d);
    mj_deleteModel(m);
    // mj_deactivate();

    return 0;
}
