#include "KinectFusionReconstruction.h"
//#define LOG_ICP

KinectFusionReconstruction::KinectFusionReconstruction() {
    depth_width = 0;
    depth_height = 0;
}

void KinectFusionReconstruction::SetYamlParameters(const YAML::Node& config_) {
    this->config = config_;
    // set TSDF volume parameter
    int resolutionX = config["tsdf_size_x"].as<int>();
    int resolutionY = config["tsdf_size_y"].as<int>();
    int resolutionZ = config["tsdf_size_z"].as<int>();
    volume_resolution = Eigen::Vector3i(resolutionX, resolutionY, resolutionZ);
    voxel_size = config["tsdf_voxel_size"].as<float>();
    max_integration_weight = config["max_integration_weight"].as<int>();
    auto thres_range = config["thres_range"].as<float>();

    // set init camera pose in world
    world2camera.setIdentity();
    //    world2camera(0, 3).imag(H_);
    world2camera_record.reserve(10000);
    world2camera_record.push_back(world2camera);
    world2volume.setIdentity();
    auto init_x = config["init_x"].as<float>();
    auto init_y = config["init_y"].as<float>();
    auto init_z = config["init_z"].as<float>();
    float r_x = config["r_x"].as<float>() / 180.0f * float(M_PI);
    float r_y = config["r_y"].as<float>() / 180.0f * float(M_PI);
    float r_z = config["r_z"].as<float>() / 180.0f * float(M_PI);
    Eigen::AngleAxisf Rx(Eigen::AngleAxisf(r_x, Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf Ry(Eigen::AngleAxisf(r_y, Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf Rz(Eigen::AngleAxisf(r_z, Eigen::Vector3f::UnitZ()));
    Eigen::Matrix3f rotation = (Rx * Ry * Rz).matrix();
    Eigen::Vector3f translation(init_x, init_y, init_z);
    world2volume.block<3, 3>(0, 0) = rotation;
    world2volume.block<3, 1>(0, 3) = translation;

    // set depth camera parameters
    depth_width = config["depth_width"].as<int>();
    depth_height = config["depth_height"].as<int>();
    kinect_intrinsic.fx = config["fx"].as<float>();
    kinect_intrinsic.fy = config["fy"].as<float>();
    kinect_intrinsic.cx = config["cx"].as<float>();
    kinect_intrinsic.cy = config["cy"].as<float>();

    // set ICP parameters
    num_levels = config["num_levels"].as<int>();
    if (num_levels > 3) {
        std::cout << "sorry, the max supported multi-level = 3"
                  << "\n";
    }
    int iters[] = {5, 4, 3};
    std::copy(iters, iters + num_levels, icp_iterations);

    distThres = config["distThres"].as<float>();
    angleThres = float(sin(config["angleThres"].as<float>() / 180.f * M_PI));

    // parameters for grad SLAM
    biInterpolate_threshold = config["biInterpolate_threshold"].as<float>();
    trunc_logistic_k = config["trunc_logistic_k"].as<float>();

    // init memory
    AllocateBuffers();
    tsdf_volume_d_ptr =
            new TsdfVolume(volume_resolution, voxel_size, thres_range);

    use_gtPose = config["flag_use_gtPose"].as<bool>();
    gt_poses.resize(0);
    frame_id = 0;
    frame_step = config["frame_step"].as<int>();
}

void KinectFusionReconstruction::AllocateBuffers() {
    depths_curr_d.resize(num_levels);
    vmaps_curr_d.resize(num_levels);
    nmaps_curr_d.resize(num_levels);
    vmaps_g_prev_d.resize(num_levels);
    nmaps_g_prev_d.resize(num_levels);
    newTSDF_volume.create(volume_resolution.y() * volume_resolution.z(),
                          volume_resolution.x());

    for (int i = 0; i < num_levels; ++i) {
        int pyr_rows = depth_height >> i;
        int pyr_cols = depth_width >> i;
        depths_curr_d[i].create(pyr_rows, pyr_cols);
        vmaps_curr_d[i].create(pyr_rows * 3, pyr_cols);
        nmaps_curr_d[i].create(pyr_rows * 3, pyr_cols);
        vmaps_g_prev_d[i].create(pyr_rows * 3, pyr_cols);
        nmaps_g_prev_d[i].create(pyr_rows * 3, pyr_cols);
    }
    // create ICP gpu buffer
    g_buf.create(27, 20 * 60);
    sum_buf.create(27);
    jacobi_buf.create(12, 20 * 60);
    jacobi_loss_pose = Eigen::Matrix4f::Zero();
    for (auto &hessian_loss_pose: hessian_loss_pose_list) {
        hessian_loss_pose = new Eigen::Matrix4f[4];
        for (int j = 0; j < 4; j++) {
            hessian_loss_pose[j] = Eigen::Matrix4f::Zero();
        }
    }
    for (auto &hessian_buf_i: hessian_buf)
        hessian_buf_i.create(12, 20 * 60);
}

void KinectFusionReconstruction::ReleaseBuffers() {
    newTSDF_volume.release();
    for (int i = 0; i < num_levels; ++i) {
        depths_curr_d[i].release();
        vmaps_curr_d[i].release();
        nmaps_curr_d[i].release();
        vmaps_g_prev_d[i].release();
        nmaps_g_prev_d[i].release();
    }
    g_buf.release();
    sum_buf.release();
    jacobi_buf.release();
    for (auto &hessian_buf_i: hessian_buf)
        hessian_buf_i.release();
    delete tsdf_volume_d_ptr;
}

int KinectFusionReconstruction::SmoothDepthFrame(
        MapArr &dst_d, const DeviceArray2D<ushort> &src_d) {
    //	check whether illegal input
    if (src_d.rows() <= 0 || src_d.cols() <= 0) {
        std::cout << "error: KinectFusionReconstruction::SmoothDepthFrame, input "
                     "map is empty"
                  << std::endl;
        return 0;
    }

    //	if dst_d size don't match src_d, resize dst_d
    if (dst_d.rows() != src_d.rows() || dst_d.cols() != src_d.cols() ||
        dst_d.step() != src_d.step()) {
        dst_d.release();
        dst_d.create(src_d.rows(), src_d.cols());
    }

    //	filter
    bilateralFilter(src_d, dst_d);
    return 1;
}

int KinectFusionReconstruction::ProcessFrame(
        const DeviceArray2D<ushort> &depth_frame_d) {
    //  map process and ICP
    int align_return = AlignDepthToReconstruction(depth_frame_d, false);
    if (frame_id > 0 && !align_return) {
        std::cout << "Frame align failed!" << std::endl;
        return 0;
    }
    //  fuse into volume
    IntegrateFrame(depth_frame_d);
    frame_id += frame_step;
    return 1;
}

int KinectFusionReconstruction::AlignDepthToReconstruction(
        const DeviceArray2D<ushort> &depth_frame_d, bool use_LM) {
    SurfaceMeasure(depth_frame_d);
    if (use_gtPose) {
        return 1;
    }
    Eigen::Matrix4cf c2w_prev = world2camera_record.back().inverse();
    Matrix3frm Rprev = GetRotation(c2w_prev);
    Eigen::Vector3cf tprev = GetTranslation(c2w_prev);
    Matrix3frm Rprev_inv = Rprev.inverse();
    Matrix3frm Rcurr = Rprev;
    Eigen::Vector3cf tcurr = tprev;
    int res = PoseEstimate(Rcurr, tcurr, Rprev_inv, tprev);
    return res;
}

int KinectFusionReconstruction::PoseEstimate(Matrix3frm Rcurr, Eigen::Vector3cf tcurr,
                                             Matrix3frm Rprev_inv, Eigen::Vector3cf tprev) {
    if (frame_id == 0) {
        return 0;
    } else {
        Eigen::Matrix4cf c2w_prev = world2camera_record.back().inverse();
        Eigen::Matrix4cf c2w_curr = c2w_prev;
        auto &device_Rprev_inv = device_cast<MatS33>(Rprev_inv);
        auto &device_tprev = device_cast<devComplex3>(tprev);
        for (int level_index = num_levels - 1; level_index >= 0; --level_index) {
            MapArr &vmap_curr = vmaps_curr_d[level_index];
            MapArr &nmap_curr = nmaps_curr_d[level_index];
            MapArr &vmap_g_prev = vmaps_g_prev_d[level_index];
            MapArr &nmap_g_prev = nmaps_g_prev_d[level_index];
            int iter_num = icp_iterations[level_index];
            for (int iter = 0; iter < iter_num; ++iter) {
                auto &device_Rcurr = device_cast<MatS33>(Rcurr);
                auto &device_tcurr = device_cast<devComplex3>(tcurr);

                Eigen::Matrix<hostComplexICP, 6, 6> A;
                Eigen::Matrix<hostComplexICP, 6, 1> b;
                estimateCombined(device_Rcurr, device_tcurr, vmap_curr, nmap_curr,
                                 device_Rprev_inv, device_tprev,
                                 kinect_intrinsic(level_index), vmap_g_prev,
                                 nmap_g_prev, distThres, angleThres, g_buf, sum_buf,
                                 A.data(), b.data());
                double det = A.real().determinant();
                if (fabs(det) < 1e-15 || isnan(det)) {
                    if (isnan(det))
                        std::cout << "qnan det" << std::endl;
                    else
                        std::cout << "eps det: " << fabs(det) << std::endl;
                    return 0;
                }
                Eigen::Matrix<std::complex<float>, 6, 1> result = A.llt().solve(b).cast<std::complex<float>>();
                std::complex<float> alpha = (std::complex<float>) result(0);
                std::complex<float> beta = (std::complex<float>) result(1);
                std::complex<float> gamma = (std::complex<float>) result(2);
                Matrix3c Rinc =
                        (Matrix3c) AngleAxisc(gamma, Vector3c::UnitZ()) *
                        AngleAxisc(beta, Vector3c::UnitY()) *
                        AngleAxisc(alpha, Vector3c::UnitX());
                Vector3c tinc = result.tail<3>().cast<hostComplex>();
                tcurr = Rinc * tcurr + tinc;
                Rcurr = Rinc * Rcurr;
                c2w_curr.block<3, 3>(0, 0) = Rcurr;
                c2w_curr.block<3, 1>(0, 3) = tcurr;
                c2w_curr(3, 3) = 1;
            }
        }
#ifdef LOG_ICP
        out << "\n";
        out.close();
#endif
        world2camera = c2w_curr.inverse();
        world2camera_record.push_back(world2camera);
        return 1;
    }
}

int KinectFusionReconstruction::IntegrateFrame(
        const DeviceArray2D<ushort> &depth_frame_d) {
    if (use_gtPose) {
        Eigen::Matrix4f gtPose_real = gt_poses[frame_id];
        Eigen::Matrix4cf gtPose;
        gtPose.setZero();
        gtPose.real() = gtPose_real;
        Eigen::Matrix4cf c2w = gtPose;
        world2camera = c2w.inverse();
        world2camera_record.back() = world2camera;
    }
    Eigen::Matrix4cf c2w = world2camera_record.back().inverse();
    Eigen::Matrix4cf c2v = world2volume * c2w;
    Eigen::Matrix4cf v2c = c2v.inverse();
    // compute pose for cuda
    Eigen::Vector3cf tc2v = GetTranslation(c2v);
    auto &device_tc2v = device_cast<devComplex3>(tc2v);

    Matrix3frm Rv2c = GetRotation(v2c);
    auto &device_Rv2c = device_cast<MatS33>(Rv2c);
    Eigen::Vector3cf tv2c = GetTranslation(v2c);
    auto &device_tv2c = device_cast<devComplex3>(tv2c);

    int3 volume_res;
    volume_res.x = volume_resolution.x();
    volume_res.y = volume_resolution.y();
    volume_res.z = volume_resolution.z();
    integrateTsdfVolume(depth_frame_d, kinect_intrinsic, max_integration_weight,
                        volume_res, voxel_size, device_Rv2c, device_tv2c, device_tc2v,
                        tsdf_volume_d_ptr->getTsdfTruncDist(),
                        tsdf_volume_d_ptr->value(), tsdf_volume_d_ptr->weight(),
                        tsdf_volume_d_ptr->grad(), depthRawScaled_d, frame_id,
                        biInterpolate_threshold, trunc_logistic_k);

    // ray casting
    CalculatePointCloud(vmaps_g_prev_d[0], nmaps_g_prev_d[0]);
    for (int i = 1; i < num_levels; ++i) {
        resizeVMap(vmaps_g_prev_d[i - 1], vmaps_g_prev_d[i]);
        resizeNMap(nmaps_g_prev_d[i - 1], nmaps_g_prev_d[i]);
    }
    return 1;
}

void KinectFusionReconstruction::SurfaceMeasure(const DeviceArray2D<ushort> &depth_frame_d) {
    if (depth_width <= 0 || depth_height <= 0) {
        std::cout << "error::KinectFusionReconstruction, not created yet"
                  << std::endl;
    } else {
        //  filter the depth
        SmoothDepthFrame(depths_curr_d[0], depth_frame_d);

        //  create pyramid
        for (int i = 1; i < num_levels; ++i)
            pyrDown(depths_curr_d[i - 1], depths_curr_d[i]);
        //  calculate point cloud and normal map
        for (int i = 0; i < num_levels; ++i) {
            createVMap(
                    kinect_intrinsic(i), depths_curr_d[i],
                    vmaps_curr_d[i]);//	camera coordinate, +z is camera direction
            createNMap(vmaps_curr_d[i], nmaps_curr_d[i]);
        }
    }
}


int KinectFusionReconstruction::CalculatePointCloud(MapArr &xyz_g_d,
                                                    MapArr &normal_g_d) {

    Eigen::Matrix4cf c2w = world2camera.inverse();
    Eigen::Matrix4cf c2v = world2volume * c2w;
    Eigen::Matrix4cf v2w = world2volume.inverse();

    Matrix3frm Rc2v = GetRotation(c2v);
    Eigen::Vector3cf tc2v = c2v.block<3, 1>(0, 3);
    Matrix3frm Rc2w = GetRotation(c2w);
    Eigen::Vector3cf tc2w = c2w.block<3, 1>(0, 3);
    Matrix3frm Rv2w = GetRotation(v2w);
    Eigen::Vector3cf tv2w = v2w.block<3, 1>(0, 3);
    auto &device_Rc2v = device_cast<MatS33>(Rc2v);
    auto &device_tc2v = device_cast<devComplex3>(tc2v);
    auto &device_Rc2w = device_cast<MatS33>(Rc2w);
    auto &device_tc2w = device_cast<devComplex3>(tc2w);
    auto &device_Rv2w = device_cast<MatS33>(Rv2w);
    auto &device_tv2w = device_cast<devComplex3>(tv2w);

    int3 volume_res;
    volume_res.x = volume_resolution.x();
    volume_res.y = volume_resolution.y();
    volume_res.z = volume_resolution.z();

    raycast(kinect_intrinsic, device_Rc2v, device_tc2v, device_Rv2w, device_tv2w,
            tsdf_volume_d_ptr->getTsdfTruncDist(), volume_res, voxel_size,
            tsdf_volume_d_ptr->value(), tsdf_volume_d_ptr->grad(), xyz_g_d,
            normal_g_d);
    return 0;
}

CPointCloud KinectFusionReconstruction::ExportPointCloud(int max_buffer) const {
    DeviceArray<float3> cloud_buffer, normal_buffer;
    cloud_buffer.create(max_buffer);
    normal_buffer.create(max_buffer);

    int3 volume_res;
    volume_res.x = volume_resolution.x();
    volume_res.y = volume_resolution.y();
    volume_res.z = volume_resolution.z();
    size_t num_points = extractPoints(
            tsdf_volume_d_ptr->value(), tsdf_volume_d_ptr->weight(),
            tsdf_volume_d_ptr->grad(), volume_res, voxel_size, cloud_buffer);
    extractNormals(tsdf_volume_d_ptr->value(), tsdf_volume_d_ptr->weight(),
                   tsdf_volume_d_ptr->grad(), volume_res, voxel_size,
                   cloud_buffer, normal_buffer);

    auto *points = new float3[num_points];
    auto *normals = new float3[num_points];
    cudaMemcpy(points, static_cast<const void *>(cloud_buffer),
               num_points * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normals, static_cast<const void *>(normal_buffer),
               num_points * sizeof(float3), cudaMemcpyDeviceToHost);

    CPointCloud res;
    for (int i = 0; i < num_points; i++) {
        Eigen::Vector3f position(points[i].x, points[i].y, points[i].z);
        Eigen::Vector3f normal(normals[i].x, normals[i].y, normals[i].z);
        Eigen::Vector3f color((normals[i].x + 1.0f) / 2.0f,
                              (normals[i].y + 1.0f) / 2.0f,
                              (normals[i].z + 1.0f) / 2.0f);
        res.addPoint(CPoint(position, normal, color));
    }

    delete[] points;
    delete[] normals;
    cloud_buffer.release();
    normal_buffer.release();
    return res;
}

// float2 KinectFusionReconstruction::ComputeTSDF_loss(const Eigen::VectorXf &xi, const DeviceArray2D<ushort> &depth_frame_d,
//                                                     thrustDvec<float> &gt_vec,
//                                                     thrustDvec<float> &real_vec,
//                                                     thrustDvec<int> &count_vec) {
//     gt_vec.clear();
//     real_vec.clear();
//     count_vec.clear();
//     Eigen::Matrix4f c2v = se3_exp(xi);
//     Eigen::Matrix4f v2c = c2v.inverse();
//     Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rv2c;
//     for (int j = 0; j < 3; j++)
//         for (int i = 0; i < 3; i++)
//             Rv2c(j, i) = c2v(i, j);
//     Eigen::Vector3f tv2c = v2c.block<3, 1>(0, 3);
//     auto &device_Rv2c = device_cast<Mat33>(Rv2c);
//     auto &device_tv2c = device_cast<float3>(tv2c);
//     // set volume res
//     int3 volume_res;
//     volume_res.x = volume_resolution.x();
//     volume_res.y = volume_resolution.y();
//     volume_res.z = volume_resolution.z();
//     float2 res = ComputeLocalTsdf_loss(depth_frame_d, kinect_intrinsic, depthRawScaled_d,
//                                        volume_res, voxel_size, device_Rv2c, device_tv2c,
//                                        tsdf_volume_d_ptr->getTsdfTruncDist(),
//                                        biInterpolate_threshold, trunc_logistic_k,
//                                        gt_vec, real_vec, count_vec);
//     return res;
// }

// float4 KinectFusionReconstruction::ComputeTSDF_hessian(const VectorXdc &xi,
//                                                        const DeviceArray2D<ushort> &depth_frame_d,
//                                                        thrustDvec<float> &gt_vec,
//                                                        thrustDvec<float> &real_vec,
//                                                        thrustDvec<float> &grad_vec,
//                                                        thrustDvec<float> &hessian_vec,
//                                                        thrustDvec<int> &count_vec) {
//     gt_vec.clear();
//     real_vec.clear();
//     grad_vec.clear();
//     hessian_vec.clear();
//     count_vec.clear();
//     // compute pose for cuda
//     Matrix4dc c2v = se3_exp_dc(xi);
//     Matrix4dc v2c = c2v.inverse();
//     Eigen::Matrix<DoubleComplex, 3, 3, Eigen::RowMajor> Rv2c;
//     for (int j = 0; j < 3; j++)
//         for (int i = 0; i < 3; i++)
//             Rv2c(j, i) = c2v(i, j);
//     Vector3dc tv2c = v2c.block<3, 1>(0, 3);
//     auto &device_Rv2c = device_cast<MatD33>(Rv2c);
//     auto &device_tv2c = device_cast<devDComplex3>(tv2c);
//     // set volume res
//     int3 volume_res;
//     volume_res.x = volume_resolution.x();
//     volume_res.y = volume_resolution.y();
//     volume_res.z = volume_resolution.z();
//     float4 res = ComputeLocalTsdf_hessian(depth_frame_d, kinect_intrinsic, depthRawScaled_d,
//                                           volume_res, voxel_size, device_Rv2c, device_tv2c,
//                                           tsdf_volume_d_ptr->getTsdfTruncDist(),
//                                           biInterpolate_threshold, trunc_logistic_k,
//                                           gt_vec, real_vec, grad_vec, hessian_vec, count_vec);
//     return res;
// }

void KinectFusionReconstruction::saveTSDFVolume(const std::string &tsdf_filename) {
    std::vector<float> tsdf_vector;
    tsdf_volume_d_ptr->downloadTSDFWithoutGrad(tsdf_vector);
    std::ofstream save_tsdf_stream(tsdf_filename, std::ios::out | std::ios::binary | std::ios::trunc);
    save_tsdf_stream.write((char *) tsdf_vector.data(),
                           volume_resolution[0] * volume_resolution[2] * volume_resolution[2] * sizeof(float));
    save_tsdf_stream.close();
    tsdf_vector.clear();
    tsdf_vector.shrink_to_fit();
}

CPointCloud KinectFusionReconstruction::generateCurrPC(int level) {
    DeviceArray2D<devComplex> v_map_d = vmaps_curr_d[level];
    std::vector<std::complex<float>> v_map;
    v_map.resize(v_map_d.cols() * v_map_d.rows());
    v_map_d.download(v_map.data(), v_map_d.cols() * sizeof(std::complex<float>));
    DeviceArray2D<devComplex> n_map_d = nmaps_curr_d[level];
    std::vector<std::complex<float>> n_map;
    n_map.resize(n_map_d.cols() * n_map_d.rows());
    n_map_d.download(n_map.data(), n_map_d.cols() * sizeof(std::complex<float>));

    std::vector<CPoint> points;
    int pyr_width = v_map_d.cols();
    int pyr_height = v_map_d.rows() / 3;
    for (int i = 0; i < pyr_height; i++) {
        for (int j = 0; j < pyr_width; j++) {
            float p_x = v_map[i * pyr_width + j].real();
            float p_y = v_map[(i + pyr_height) * pyr_width + j].real();
            float p_z = v_map[(i + pyr_height * 2) * pyr_width + j].real();
            Eigen::Vector3f position(p_x, p_y, p_z);
            float n_x = n_map[i * pyr_width + j].real();
            float n_y = n_map[(i + pyr_height) * pyr_width + j].real();
            float n_z = n_map[(i + pyr_height * 2) * pyr_width + j].real();
            Eigen::Vector3f normal(n_x, n_y, n_z);
            Eigen::Vector4f color(n_x, n_y, n_z, 1);
            points.emplace_back(position, normal, color);
        }
    }
    CPointCloud point_cloud(points);
    return point_cloud;
}

CPointCloud KinectFusionReconstruction::generatePrevPC(int level) {
    DeviceArray2D<devComplex> v_map_d = vmaps_g_prev_d[level];
    std::vector<std::complex<float>> v_map;
    v_map.resize(v_map_d.cols() * v_map_d.rows());
    v_map_d.download(v_map.data(), v_map_d.cols() * sizeof(std::complex<float>));
    DeviceArray2D<devComplex> n_map_d = nmaps_g_prev_d[level];
    std::vector<std::complex<float>> n_map;
    n_map.resize(n_map_d.cols() * n_map_d.rows());
    n_map_d.download(n_map.data(), n_map_d.cols() * sizeof(std::complex<float>));
    std::vector<CPoint> points;
    int pyr_width = v_map_d.cols();
    int pyr_height = v_map_d.rows() / 3;
    for (int i = 0; i < pyr_height; i++) {
        for (int j = 0; j < pyr_width; j++) {
            float p_x = v_map[i * pyr_width + j].real();
            float p_y = v_map[(i + pyr_height) * pyr_width + j].real();
            float p_z = v_map[(i + pyr_height * 2) * pyr_width + j].real();
            Eigen::Vector3f position(p_x, p_y, p_z);
            float n_x = n_map[i * pyr_width + j].real();
            float n_y = n_map[(i + pyr_height) * pyr_width + j].real();
            float n_z = n_map[(i + pyr_height * 2) * pyr_width + j].real();
            Eigen::Vector3f normal(n_x, n_y, n_z);
            Eigen::Vector3f color(n_x, n_y, n_z);
            points.emplace_back(position, normal, color);
        }
    }
    CPointCloud point_cloud(points);
    return point_cloud;
}
